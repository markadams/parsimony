/*************************************************************************************
    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/debug/Test_hdcg_parsimony.cc

    Copyright (C) 2023

Author: Peter Boyle <pboyle@bnl.gov>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */

/* Storage-precision instrumentation of Test_general_coarse_hdcg.cc.
 *
 * Identical physics to that driver.  Three preconditioner inputs can be
 * rounded to a reduced storage format before they are used, arithmetic stays
 * fp64 throughout, and the result is written as JSON so a search can drive it:
 *
 *   --round-subspace <fmt>   near-null vectors, after orthogonalisation and
 *                            before the coarse operator is built from them
 *   --round-stencil  <fmt>   the coarse stencil blocks, after coarsening
 *   --round-evec     <fmt>   the coarse deflation eigenvectors, before import
 *
 * <fmt> is v1..v53 (float64 exponent, k significand bits) or a named format
 * (fp32, tf32, bf16, fp16, fp8e4m3, fp8e5m2).  v53 is rounding off.
 *
 * The fine operator is never rounded, so the true residual that TwoLevelCGmrhs
 * recomputes after the solve remains an independent fp64 check on a solver
 * whose preconditioner has been degraded.
 *
 * See casestudies/hdcg/{PLAN,NOTES,DRIVER-PATCH}.md in the parsimony repo.
 */

#include <Grid/Grid.h>
#include <Grid/algorithms/iterative/ImplicitlyRestartedBlockLanczos.h>
#include <Grid/algorithms/iterative/ImplicitlyRestartedBlockLanczosCoarse.h>
#include <Grid/algorithms/iterative/AdefMrhs.h>

#include "round_field.h"

#include <chrono>
#include <cstdio>
#include <map>
#include <string>

#ifndef PARSIMONY_NBASIS
#define PARSIMONY_NBASIS 40
#endif

using namespace std;
using namespace Grid;

static double now()
{
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Read a rounding format from the command line.  An unrecognised name aborts:
// silently running at fp64 would produce a data point labelled bf16.
static Parsimony::RoundSpec ReadSpec(int argc, char **argv, const std::string &opt)
{
  Parsimony::RoundSpec spec = Parsimony::ROUND_OFF;
  if (GridCmdOptionExists(argv, argv + argc, opt)) {
    std::string arg = GridCmdOptionPayload(argv, argv + argc, opt);
    if (!Parsimony::parse_round_spec(arg.c_str(), spec)) {
      std::cout << GridLogError << "unrecognised format '" << arg << "' for " << opt
                << " (expected v1..v53, fp32, tf32, bf16, fp16, fp8e4m3, fp8e5m2)"
                << std::endl;
      GRID_ASSERT(0);
    }
  }
  return spec;
}

static std::string SpecName(int argc, char **argv, const std::string &opt)
{
  if (GridCmdOptionExists(argv, argv + argc, opt))
    return GridCmdOptionPayload(argv, argv + argc, opt);
  return "v53";
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  ////////////////////////// Parsimony knobs ///////////////////////////////
  Parsimony::RoundSpec spec_subspace = ReadSpec(argc,argv,"--round-subspace");
  Parsimony::RoundSpec spec_stencil  = ReadSpec(argc,argv,"--round-stencil");
  Parsimony::RoundSpec spec_evec     = ReadSpec(argc,argv,"--round-evec");

  int case_id = 0;
  if ( GridCmdOptionExists(argv,argv+argc,"--case") ){
    // GridCmdOptionInt takes a non-const reference, so it cannot bind a temporary.
    std::string arg = GridCmdOptionPayload(argv,argv+argc,"--case");
    GridCmdOptionInt(arg,case_id);
  }
  std::string json_path;
  if ( GridCmdOptionExists(argv,argv+argc,"--json") ){
    json_path = GridCmdOptionPayload(argv,argv+argc,"--json");
  }
  std::string gauge("ckpoint_lat.4000");
  if ( GridCmdOptionExists(argv,argv+argc,"--gauge") ){
    gauge = GridCmdOptionPayload(argv,argv+argc,"--gauge");
  }

  // Chebyshev order for the subspace filter.  500 is the physics value and the
  // stock driver's; lowering it makes setup cheap when the thing being
  // debugged is downstream of the subspace rather than the subspace itself.
  // A run that lowers it is not a measurement.
  int cheby_ord = 500;
  if ( GridCmdOptionExists(argv,argv+argc,"--cheby-ord") ){
    std::string arg = GridCmdOptionPayload(argv,argv+argc,"--cheby-ord");
    GridCmdOptionInt(arg,cheby_ord);
    if ( cheby_ord != 500 ) {
      std::cout << GridLogMessage << "PARSIMONY WARNING: Chebyshev order "
		<< cheby_ord << " != 500, this run is a debug run and its "
		<< "iteration counts are not a measurement" << std::endl;
    }
  }

  // How the near-null subspace is built.  "chebyshev" is the physics choice and
  // the stock driver's.  "random" is Grid's own CreateSubspaceRandom, as used by
  // tests/debug/Test_general_coarse.cc: a weak coarse space that converges
  // slowly, which is useless physics but a well-posed and much better
  // conditioned test of the precision machinery -- a large reference iteration
  // count is what gives the iteration-budget criterion room to resolve a knee.
  std::string subspace_kind("chebyshev");
  if ( GridCmdOptionExists(argv,argv+argc,"--subspace") ){
    subspace_kind = GridCmdOptionPayload(argv,argv+argc,"--subspace");
  }

  std::map<std::string,double> delta;   // relative L2 change each rounding caused

  const int Ls=16;
  // The padded coarse operator costs npoint * padded_sites * nbasis^2 complex
  // doubles, and at nbasis 40 that is ~16 GB per rank on this lattice -- more
  // than an A100 has, at any rank count, because the depth-3 halo dominates a
  // 512-site coarse grid.  Build with -DPARSIMONY_NBASIS=20 to cut it 4x.
  const int nbasis = PARSIMONY_NBASIS;
  const int cb = 0 ;
  RealD mass=0.01;
  RealD M5=1.8;
  RealD b=1.0;
  RealD c=0.0;

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
								   GridDefaultSimd(Nd,vComplex::Nsimd()),
								   GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  // Construct a coarsened grid with 4^4 cell
  Coordinate Block({4,4,4,4});
  Coordinate clatt = GridDefaultLatt();
  for(int d=0;d<clatt.size();d++){
    clatt[d] = clatt[d]/Block[d];
  }

  GridCartesian *Coarse4d =  SpaceTimeGrid::makeFourDimGrid(clatt,
							    GridDefaultSimd(Nd,vComplex::Nsimd()),
							    GridDefaultMpi());;
  GridCartesian *Coarse5d =  SpaceTimeGrid::makeFiveDimGrid(1,Coarse4d);

  ///////////////////////// RNGs /////////////////////////////////
  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  std::vector<int> cseeds({5,6,7,8});

  GridParallelRNG          RNG5(FGrid);   RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG          RNG4(UGrid);   RNG4.SeedFixedIntegers(seeds4);
  GridParallelRNG          CRNG(Coarse5d);CRNG.SeedFixedIntegers(cseeds);

  ///////////////////////// Configuration /////////////////////////////////
  LatticeGaugeField Umu(UGrid);

  if ( gauge == std::string("free") ) {
    // Unit gauge: the free Dirac operator.  The physics is trivial but the
    // linear algebra is not -- the aggregation, the coarse stencil and the
    // deflation all do real work -- so this exercises the whole pipeline with
    // no external input.  Bit-knees measured here are preliminary and belong
    // to the free field, not to a gauge ensemble; label every one of them.
    std::cout << GridLogMessage << "PARSIMONY WARNING: free field (unit gauge), "
	      << "knees are preliminary and not an ensemble result" << std::endl;
    SU<Nc>::ColdConfiguration(Umu);
  } else if ( gauge == std::string("hot") ) {
    // A hot configuration has no low-mode structure at all; weaker still.
    std::cout << GridLogMessage << "PARSIMONY WARNING: hot configuration, "
	      << "results validate the pipeline and nothing else" << std::endl;
    SU<Nc>::HotConfiguration(RNG4,Umu);
  } else {
    FieldMetaData header;
    NerscIO::readConfiguration(Umu,header,gauge);
  }

  //////////////////////// Fermion action //////////////////////////////////
  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c);

  SchurDiagMooeeOperator<MobiusFermionD, LatticeFermion> HermOpEO(Ddwf);

  typedef HermOpAdaptor<LatticeFermionD> HermFineMatrix;
  HermFineMatrix FineHermOp(HermOpEO);

  LatticeFermionD src(FrbGrid);
  src = ComplexD(1.0);
  PowerMethod<LatticeFermionD>       PM;   PM(HermOpEO,src);

  ////////////////////////////////////////////////////////////
  ///////////// Coarse basis and Little Dirac Operator ///////
  ////////////////////////////////////////////////////////////
  typedef GeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nbasis> LittleDiracOperator;
  typedef LittleDiracOperator::CoarseVector CoarseVector;

  NextToNextToNextToNearestStencilGeometry5D geom(Coarse5d);

  typedef Aggregation<vSpinColourVector,vTComplex,nbasis> Subspace;
  Subspace Aggregates(Coarse5d,FrbGrid,cb);

  const double t_setup_start = now();

  int refine=1;
  std::cout << "**************************************"<<std::endl;
  std::cout << "Create Subspace"<<std::endl;
  std::cout << "**************************************"<<std::endl;
  if ( subspace_kind == std::string("random") ) {
    std::cout << GridLogMessage << "PARSIMONY WARNING: random subspace, not a "
	      << "physics coarse space; iteration counts are a property of the "
	      << "harness, and only ratios between them mean anything"
	      << std::endl;
    Aggregates.CreateSubspaceRandom(RNG5);
  } else {
    Aggregates.CreateSubspaceChebyshev(RNG5,HermOpEO,nbasis,35.,0.01,cheby_ord);
  }

  std::cout << "**************************************"<<std::endl;
  std::cout << "Coarsen after refine"<<std::endl;
  std::cout << "**************************************"<<std::endl;
  Aggregates.Orthogonalise();

  ////////////////////////////////////////////////////////////////////////
  // HOOK 1: subspace.
  //
  // Before CoarsenOperator, so the coarse operator is built from the same
  // rounded vectors the projector later imports: the two-level method stays
  // Galerkin-consistent and what is measured is the subspace's own bit
  // requirement.  Orthogonality is degraded to O(2^-k) here; that is part of
  // the measurement, not a defect to repair with a second Orthogonalise().
  ////////////////////////////////////////////////////////////////////////
  if ( !spec_subspace.identity() ) {
    delta["subspace"] = Parsimony::RoundFields(Aggregates.subspace,spec_subspace);
    std::cout << GridLogMessage << "PARSIMONY rounded subspace, relative L2 change "
	      << delta["subspace"] << std::endl;
  }

  std::cout << "**************************************"<<std::endl;
  std::cout << "Building MultiRHS Coarse operator"<<std::endl;
  std::cout << "**************************************"<<std::endl;
  ConjugateGradient<CoarseVector>  coarseCG(4.0e-2,20000,true);

  const int nrhs=8;

  Coordinate mpi=GridDefaultMpi();
  Coordinate rhMpi ({1,1,mpi[0],mpi[1],mpi[2],mpi[3]});
  Coordinate rhLatt({nrhs,1,clatt[0],clatt[1],clatt[2],clatt[3]});
  Coordinate rhSimd({vComplex::Nsimd(),1, 1,1,1,1});

  GridCartesian *CoarseMrhs = new GridCartesian(rhLatt,rhSimd,rhMpi);
  typedef MultiGeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nbasis> MultiGeneralCoarsenedMatrix_t;
  MultiGeneralCoarsenedMatrix_t mrhs(geom,CoarseMrhs);

  mrhs.CoarsenOperator(FineHermOp,Aggregates,Coarse5d);

  ////////////////////////////////////////////////////////////////////////
  // HOOK 2: coarse stencil.
  //
  // GetMatrix/SetMatrix round-trip each geometry point through a Lattice on
  // the single-rhs coarse grid.  Unlike hook 1 this deliberately breaks
  // consistency between the coarse operator and the projector, which is why
  // it is expected to knee more sharply.
  ////////////////////////////////////////////////////////////////////////
  if ( !spec_stencil.identity() ) {
    typedef MultiGeneralCoarsenedMatrix_t::CoarseMatrix CoarseMatrix;
    const int npoint = mrhs.Geometry().npoint;
    std::vector<CoarseMatrix> Astencil(npoint,CoarseMatrix(Coarse5d));
    double worst = 0.0;
    for(int p=0;p<npoint;p++){
      mrhs.GetMatrix(p,Astencil);
      const double d = Parsimony::RoundField(Astencil[p],spec_stencil);
      if ( d > worst ) worst = d;
      mrhs.SetMatrix(p,Astencil);
    }
    delta["stencil"] = worst;
    std::cout << GridLogMessage << "PARSIMONY rounded stencil over "<<npoint
	      << " points, worst relative L2 change " << worst << std::endl;
  }

  std::cout << "**************************************"<<std::endl;
  std::cout << "         Coarse Lanczos               "<<std::endl;
  std::cout << "**************************************"<<std::endl;

  typedef HermitianLinearOperator<MultiGeneralCoarsenedMatrix_t,CoarseVector> MrhsHermMatrix;
  Chebyshev<CoarseVector>      IRLCheby(0.01,16.0,201);  // 1 iter
  MrhsHermMatrix MrhsCoarseOp     (mrhs);

  CoarseVector pm_src(CoarseMrhs);
  pm_src = ComplexD(1.0);
  PowerMethod<CoarseVector>       cPM;
  cPM(MrhsCoarseOp,pm_src);

  int Nk=32;
  int Nm=128;
  int Nstop=Nk;
  int Nconv_test_interval=1;

  ImplicitlyRestartedBlockLanczosCoarse<CoarseVector> IRL(MrhsCoarseOp,
							  Coarse5d,
							  CoarseMrhs,
							  nrhs,
							  IRLCheby,
							  Nstop,
							  Nconv_test_interval,
							  nrhs,
							  Nk,
							  Nm,
							  1e-4,100);

  int Nconv;
  std::vector<RealD>            eval(Nm);
  std::vector<CoarseVector>     evec(Nm,Coarse5d);
  std::vector<CoarseVector>     c_src(nrhs,Coarse5d);

  std::cout << "**************************************"<<std::endl;
  std::cout << "Calling mRHS HDCG"<<std::endl;
  std::cout << "**************************************"<<std::endl;
  MultiRHSBlockProject<LatticeFermionD> MrhsProjector;
  MrhsProjector.Allocate(nbasis,FrbGrid,Coarse5d);
  MrhsProjector.ImportBasis(Aggregates.subspace);

  std::cout << "**************************************"<<std::endl;
  std::cout << " Recompute coarse evecs  "<<std::endl;
  std::cout << "**************************************"<<std::endl;
  for(int r=0;r<nrhs;r++){
    random(CRNG,c_src[r]);
  }

  IRL.calc(eval,evec,c_src,Nconv,LanczosType::irbl);

  ////////////////////////////////////////////////////////////////////////
  // HOOK 3: deflation eigenvectors.
  //
  // These only seed the coarse solve's initial guess, so this is the
  // tolerance control of the three: if evec does not go lower than subspace
  // and stencil, suspect the harness before the physics.
  ////////////////////////////////////////////////////////////////////////
  if ( !spec_evec.identity() ) {
    delta["evec"] = Parsimony::RoundFields(evec,spec_evec);
    std::cout << GridLogMessage << "PARSIMONY rounded evecs, relative L2 change "
	      << delta["evec"] << std::endl;
  }

  std::cout << "**************************************"<<std::endl;
  std::cout << " Reimport coarse evecs "<<evec.size()<<" "<<eval.size()<<std::endl;
  std::cout << "**************************************"<<std::endl;
  MultiRHSDeflation<CoarseVector> MrhsGuesser;
  MrhsGuesser.ImportEigenBasis(evec,eval);

  //////////////////////////
  // Extra HDCG parameters
  //////////////////////////
  int maxit=3000;
  ConjugateGradient<CoarseVector>  CG(5.0e-2,maxit,false);
  RealD lo=0.2;
  int ord = 7;

  DoNothingGuesser<CoarseVector> DoNothing;
  HPDSolver<CoarseVector> HPDSolveMrhs(MrhsCoarseOp,CG,DoNothing);

  /////////////////////////////////////////////////
  // Mirs smoother
  /////////////////////////////////////////////////
  RealD MirsShift = lo;
  ShiftedHermOpLinearOperator<LatticeFermionD> ShiftedFineHermOp(HermOpEO,MirsShift);
  CGSmoother<LatticeFermionD> CGsmooth(ord,ShiftedFineHermOp) ;

  MrhsADEF2Preconditioner<LatticeFermion,CoarseVector>
    ADEF2(FineHermOp, CGsmooth, HPDSolveMrhs, HPDSolveMrhs, MrhsProjector, MrhsGuesser, CoarseMrhs);
  TwoLevelCGmrhs<LatticeFermion> HDCGmrhs(1.0e-8, 500, FineHermOp, ADEF2, FrbGrid);

  const double t_setup = now() - t_setup_start;

  std::vector<LatticeFermionD> src_mrhs(nrhs,FrbGrid);
  std::vector<LatticeFermionD> res_mrhs(nrhs,FrbGrid);

  // Case 0 leaves RNG5 exactly where the unmodified driver leaves it, so an
  // all-v53 run of this binary reproduces Test_general_coarse_hdcg's iteration
  // count exactly and the sanity gate is an equality rather than a judgement.
  // Other cases reseed, which changes only the right hand sides: the gauge
  // field, the subspace and every solver parameter are already fixed above.
  if ( case_id != 0 ) {
    std::vector<int> case_seeds({5+1000*case_id,6+1000*case_id,
				 7+1000*case_id,8+1000*case_id});
    RNG5.SeedFixedIntegers(case_seeds);
  }
  for(int r=0;r<nrhs;r++){
    random(RNG5,src_mrhs[r]);
    res_mrhs[r]=Zero();
  }

  const double t_solve_start = now();
  HDCGmrhs(src_mrhs,res_mrhs);
  const double t_solve = now() - t_solve_start;

  //////////////////////////////////////////////////////////////////////////
  // Result record.  Written even when the solve fails: non-convergence is a
  // data point, a missing file is a broken run, and the evaluator tells them
  // apart.  Rank 0 only.
  //////////////////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << "PARSIMONY result"
	    << " converged " << HDCGmrhs.ConvergedFlag
	    << " iterations " << HDCGmrhs.IterationsToComplete
	    << " true_resid " << HDCGmrhs.TrueResidual
	    << " t_setup " << t_setup
	    << " t_solve " << t_solve << std::endl;

  if ( json_path.size() && UGrid->IsBoss() ) {
    FILE *fp = fopen(json_path.c_str(),"w");
    GRID_ASSERT(fp!=NULL);
    fprintf(fp,"{\"converged\": %s, \"iterations\": %d, \"true_resid\": %.17g,\n",
	    HDCGmrhs.ConvergedFlag ? "true" : "false",
	    (int)HDCGmrhs.IterationsToComplete,
	    (double)HDCGmrhs.TrueResidual);
    fprintf(fp," \"t_setup\": %.6f, \"t_solve\": %.6f, \"case\": %d,\n",
	    t_setup,t_solve,case_id);
    // Self-describing: these knobs change what the numbers mean, and a record
    // that does not carry them cannot be told apart from a physics run later.
    fprintf(fp," \"gauge\": \"%s\", \"subspace\": \"%s\", \"nbasis\": %d, \"cheby_ord\": %d,\n",
	    gauge.c_str(),subspace_kind.c_str(),nbasis,cheby_ord);
    fprintf(fp," \"round\": {\"subspace\": \"%s\", \"stencil\": \"%s\", \"evec\": \"%s\"},\n",
	    SpecName(argc,argv,"--round-subspace").c_str(),
	    SpecName(argc,argv,"--round-stencil").c_str(),
	    SpecName(argc,argv,"--round-evec").c_str());
    fprintf(fp," \"delta\": {");
    int n=0;
    for(std::map<std::string,double>::const_iterator it=delta.begin();it!=delta.end();++it){
      fprintf(fp,"%s\"%s\": %.17g",n++ ? ", " : "",it->first.c_str(),it->second);
    }
    fprintf(fp,"}\n}\n");
    fclose(fp);
    std::cout << GridLogMessage << "PARSIMONY wrote " << json_path << std::endl;
  }

  Grid_finalize();
  return 0;
}
