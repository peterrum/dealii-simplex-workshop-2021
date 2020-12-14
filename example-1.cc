// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

// Example 1: solving a Poisson problem on a mesh with a single reference-cell
// type (either hypercube or simplex).

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/numerics/data_out.h>

// for hex mesh
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q_generic.h>

// for simplex mesh
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/quadrature_lib.h>

// operators
#include "util.h"

using namespace dealii;

static unsigned int counter = 0;

template <typename PoissonOperator>
void
test(const unsigned int degree, const std::string &file_name)
{
  const int dim      = PoissonOperator::dim;
  const int spacedim = PoissonOperator::spacedim;

  parallel::shared::Triangulation<dim, spacedim> tria(MPI_COMM_WORLD);
  GridIn<dim, spacedim>(tria).read(file_name);

  const auto refere_cell_types = tria.get_reference_cell_types();

  AssertDimension(refere_cell_types.size(), 1);

  std::unique_ptr<Mapping<dim, spacedim>>       mapping;
  std::unique_ptr<FiniteElement<dim, spacedim>> fe;
  std::unique_ptr<Quadrature<dim>>              quadrature;

  if (refere_cell_types[0] == ReferenceCell::get_hypercube(dim))
    {
      mapping    = std::make_unique<MappingQGeneric<dim, spacedim>>(1);
      fe         = std::make_unique<FE_Q<dim, spacedim>>(degree);
      quadrature = std::make_unique<QGauss<dim>>(degree + 1);
    }
  else if (refere_cell_types[0] == ReferenceCell::get_simplex(dim))
    {
      mapping    = std::make_unique<MappingFE<dim, spacedim>>(Simplex::FE_P<dim, spacedim>(1));
      fe         = std::make_unique<Simplex::FE_P<dim, spacedim>>(degree);
      quadrature = std::make_unique<Simplex::QGauss<dim>>(degree + 1);
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(*fe);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
  constraints.close();

  LinearAlgebra::distributed::Vector<double> x, b;

  PoissonOperator poisson_operator(*mapping, dof_handler, constraints, *quadrature, x, b);

  ReductionControl                                     reduction_control;
  SolverCG<LinearAlgebra::distributed::Vector<double>> solver(reduction_control);
  solver.solve(poisson_operator, x, b, PreconditionIdentity());

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    printf("Solved in %d iterations.\n", reduction_control.last_step());

  constraints.distribute(x);

  DataOutBase::VtkFlags flags;

  if (refere_cell_types[0] == ReferenceCell::get_hypercube(dim)) // TODO: only working for hypercube mesh
    flags.write_higher_order_cells = true;                       //

  DataOut<dim> data_out;
  data_out.set_flags(flags);
  data_out.attach_dof_handler(dof_handler);
  x.update_ghost_values();
  data_out.add_data_vector(dof_handler, x, "solution");
  data_out.build_patches(*mapping, 2);
  data_out.write_vtu_with_pvtu_record("./", "example-1", counter++, MPI_COMM_WORLD, 2);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  for (unsigned int degree = 1; degree <= 2; ++degree)
    {
      test<PoissonOperatorMatrixBased<2>>(degree, SOURCE_DIR "/mesh/box_2D_quad.msh");
      test<PoissonOperatorMatrixBased<2>>(degree, SOURCE_DIR "/mesh/box_2D_tri.msh");

      test<PoissonOperatorMatrixFree<2>>(degree, SOURCE_DIR "/mesh/box_2D_quad.msh");
      test<PoissonOperatorMatrixFree<2>>(degree, SOURCE_DIR "/mesh/box_2D_tri.msh");
    }
}
