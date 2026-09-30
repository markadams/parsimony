/*************************************************************************************
    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/debug/Test_coarse_bits.cc

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

/* How many bits does Grid's coarse Dirac stencil need, in storage?
 *
 * Structure follows tests/debug/Test_general_coarse.cc, which is the path on
 * develop that still runs: a random aggregation, the single-RHS
 * GeneralCoarsenedMatrix, no multi-RHS class.  (Test_general_coarse_hdcg.cc
 * needs MultiGeneralCoarsenedMatrix, which lives in multigrid/deprecated/,
 * is #if 0'd out of Grid's own test, and segfaults on construction here.)
 *
 * Two coarse operators are built from the same aggregation.  One has its
 * stored stencil rounded to a reduced format; the other is kept exact.  CG
 * runs against the rounded one, so what is measured is:
 *
 *   iterations   -- the rate cost of a degraded operator
 *   true_resid   -- || b - A_exact x || / || b ||, the accuracy floor that
 *                   rounding imposes.  Judged with the EXACT operator, which
 *                   is why the exact copy is kept: the rounded operator's own
 *                   residual converges to its own, wrong, answer.
 *
 * Arithmetic is fp64 throughout; only storage is rounded.
 *
 * See casestudies/hdcg/NOTES.md in the parsimony repo.
 */

#include <Grid/Grid.h>

#include "round_field.h"

#include <chrono>
#include <cstdio>
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

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  Parsimony::RoundSpec spec = Parsimony::ROUND_OFF;
  std::string spec_name("v53");
  if ( GridCmdOptionExists(argv,argv+argc,"--round-stencil") ){
    spec_name = GridCmdOptionPayload(argv,argv+argc,"--round-stencil");
    if ( !Parsimony::parse_round_spec(spec_name.c_str(),spec) ) {
      std::cout << GridLogError << "unrecognised format '" << spec_name << "'" << std::endl;
      GRID_ASSERT(0);
    }
  }
  int case_id = 0;
  if ( GridCmdOptionExists(argv,argv+argc,"--case") ){
    std::string arg = GridCmdOptionPayload(argv,argv+argc,"--case");
    GridCmdOptionInt(arg,case_id);
  }
  std::string json_path;
  if ( GridCmdOptionExists(argv,argv+argc,"--json") ){
    json_path = GridCmdOptionPayload(argv,argv+argc,"--json");
  }
  std::string gauge("free");
  if ( GridCmdOptionExists(argv,argv+argc,"--gauge") ){
    gauge = GridCmdOptionPayload(argv,argv+argc,"--gauge");
  }

  // Ls and nbasis set the setup cost, which dominates a sweep: CoarsenOperator
  // runs once per basis vector and each application costs O(Ls).  Lowering
  // them makes a ladder affordable and makes the run preliminary; both are
  // recorded in the JSON so a cheap run cannot be mistaken for a physics one.
  int Ls=16;
  if ( GridCmdOptionExists(argv,argv+argc,"--Ls") ){
    std::string arg = GridCmdOptionPayload(argv,argv+argc,"--Ls");
    GridCmdOptionInt(arg,Ls);
  }
  const int nbasis = PARSIMONY_NBASIS;
  const int cb = 0 ;
  RealD mass=0.1;
  RealD M5=1.8;

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
								   GridDefaultSimd(Nd,vComplex::Nsimd()),
								   GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  Coordinate Block({4,4,4,4});
  Coordinate clatt = GridDefaultLatt();
  for(int d=0;d<clatt.size();d++) clatt[d] = clatt[d]/Block[d];

  GridCartesian *Coarse4d = SpaceTimeGrid::makeFourDimGrid(clatt,
							   GridDefaultSimd(Nd,vComplex::Nsimd()),
							   GridDefaultMpi());
  GridCartesian *Coarse5d = SpaceTimeGrid::makeFiveDimGrid(1,Coarse4d);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  std::vector<int> cseeds({5,6,7,8});
  GridParallelRNG RNG5(FGrid);    RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG RNG4(UGrid);    RNG4.SeedFixedIntegers(seeds4);
  GridParallelRNG CRNG(Coarse5d); CRNG.SeedFixedIntegers(cseeds);

  LatticeGaugeField Umu(UGrid);
  if ( gauge == std::string("free") ) {
    std::cout << GridLogMessage << "PARSIMONY WARNING: free field (unit gauge); "
	      << "bit counts are preliminary and not an ensemble result" << std::endl;
    SU<Nc>::ColdConfiguration(Umu);
  } else if ( gauge == std::string("hot") ) {
    std::cout << GridLogMessage << "PARSIMONY WARNING: hot configuration" << std::endl;
    SU<Nc>::HotConfiguration(RNG4,Umu);
  } else {
    FieldMetaData header;
    NerscIO::readConfiguration(Umu,header,gauge);
  }

  DomainWallFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5);
  MdagMLinearOperator<DomainWallFermionD,LatticeFermion> HermDefOp(Ddwf);

  const double t_setup_start = now();

  // Random aggregation, as Test_general_coarse.cc uses.  It is a weak coarse
  // space, which is the point: CG then takes enough iterations for the
  // iteration count to resolve anything.
  typedef Aggregation<vSpinColourVector,vTComplex,nbasis> Subspace;
  Subspace Aggregates(Coarse5d,FGrid,cb);
  Aggregates.CreateSubspaceRandom(RNG5);

  typedef GeneralCoarsenedMatrix<vSpinColourVector,vTComplex,nbasis> LittleDiracOperator;
  typedef LittleDiracOperator::CoarseVector CoarseVector;

  NextToNextToNextToNearestStencilGeometry5D geom(Coarse5d);
  HermOpAdaptor<LatticeFermionD> HOA(HermDefOp);

  // Two coarsenings of the same operator from the same aggregation: one to
  // degrade, one to judge with.
  LittleDiracOperator LittleDiracOp (geom,FGrid,Coarse5d);
  LittleDiracOperator LittleDiracRef(geom,FGrid,Coarse5d);
  LittleDiracOp .CoarsenOperator(HOA,Aggregates);
  LittleDiracRef.CoarsenOperator(HOA,Aggregates);

  const double t_setup = now() - t_setup_start;

  //////////////////////////////////////////////////////////////////////////
  // Round the stored stencil.  Both _A and _Adag: they are both stored
  // arrays, so a storage-format claim has to cover both.
  //////////////////////////////////////////////////////////////////////////
  double delta = 0.0;
  if ( !spec.identity() ) {
    const int npoint = geom.npoint;
    for(int p=0;p<npoint;p++){
      const double d = Parsimony::RoundField(LittleDiracOp._A[p],spec);
      if ( d > delta ) delta = d;
    }
    // _Adag is only allocated for a non-Hermitian operator; for a Hermitian
    // one Mdag reads _A, so rounding _A alone already covers all the stored
    // stencil there is.  Indexing _Adag unconditionally reads an empty vector.
    for(size_t p=0;p<LittleDiracOp._Adag.size();p++){
      const double d = Parsimony::RoundField(LittleDiracOp._Adag[p],spec);
      if ( d > delta ) delta = d;
    }
    std::cout << GridLogMessage << "PARSIMONY rounded stencil to " << spec_name
	      << " over " << npoint << " points (+"
	      << LittleDiracOp._Adag.size() << " dag), worst relative L2 change "
	      << delta << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////
  // CG against the rounded operator.
  //////////////////////////////////////////////////////////////////////////
  typedef HermitianLinearOperator<LittleDiracOperator,CoarseVector> HermMatrix;
  HermMatrix CoarseOp   (LittleDiracOp);
  HermMatrix CoarseOpRef(LittleDiracRef);

  if ( case_id != 0 ) {
    std::vector<int> case_seeds({5+1000*case_id,6+1000*case_id,
				 7+1000*case_id,8+1000*case_id});
    CRNG.SeedFixedIntegers(case_seeds);
  }
  CoarseVector c_src(Coarse5d), c_res(Coarse5d);
  random(CRNG,c_src);
  c_res = Zero();

  ConjugateGradient<CoarseVector> CG(1.0e-8,10000,false);
  const double t_solve_start = now();
  CG(CoarseOp,c_src,c_res);
  const double t_solve = now() - t_solve_start;

  //////////////////////////////////////////////////////////////////////////
  // Judge with the exact operator.  The rounded operator converges to its
  // own answer, so its own residual says nothing about accuracy.
  //////////////////////////////////////////////////////////////////////////
  CoarseVector c_chk(Coarse5d), c_err(Coarse5d);
  CoarseOpRef.HermOp(c_res,c_chk);
  c_err = c_chk - c_src;
  const RealD true_resid = sqrt(norm2(c_err)/norm2(c_src));

  std::cout << GridLogMessage << "PARSIMONY result"
	    << " stencil " << spec_name
	    << " iterations " << CG.IterationsToComplete
	    << " cg_resid " << CG.TrueResidual
	    << " true_resid " << true_resid
	    << " delta " << delta << std::endl;

  if ( json_path.size() && UGrid->IsBoss() ) {
    FILE *fp = fopen(json_path.c_str(),"w");
    GRID_ASSERT(fp!=NULL);
    fprintf(fp,"{\"converged\": %s, \"iterations\": %d,\n",
	    (CG.IterationsToComplete < 10000) ? "true" : "false",
	    (int)CG.IterationsToComplete);
    fprintf(fp," \"true_resid\": %.17g, \"cg_resid\": %.17g,\n",
	    (double)true_resid,(double)CG.TrueResidual);
    fprintf(fp," \"t_setup\": %.6f, \"t_solve\": %.6f, \"case\": %d,\n",
	    t_setup,t_solve,case_id);
    fprintf(fp," \"gauge\": \"%s\", \"nbasis\": %d, \"Ls\": %d, \"subspace\": \"random\",\n",
	    gauge.c_str(),nbasis,Ls);
    fprintf(fp," \"round\": {\"stencil\": \"%s\"},\n",spec_name.c_str());
    if ( spec.identity() ) fprintf(fp," \"delta\": {}\n}\n");
    else                   fprintf(fp," \"delta\": {\"stencil\": %.17g}\n}\n",delta);
    fclose(fp);
    std::cout << GridLogMessage << "PARSIMONY wrote " << json_path << std::endl;
  }

  Grid_finalize();
  return 0;
}
