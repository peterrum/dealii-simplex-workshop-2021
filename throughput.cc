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

// Throughput: TODO

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/quadrature_lib.h>

// operators
#include "util.h"

using namespace dealii;

template <typename PoissonOperator>
void
test(const unsigned int degree, const bool do_simplex, const std::vector<unsigned int> &subdivisions)
{
  const int dim      = PoissonOperator::dim;
  const int spacedim = PoissonOperator::spacedim;

  parallel::fullydistributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  // create triangulation
  {
    Triangulation<dim> tria_serial;

    const Point<dim> p1;
    const Point<dim> p2 = dim == 2 ? Point<dim>(1, 1) : Point<dim>(1, 1, 1);

    if (do_simplex)
      {
        Triangulation<dim> tria_temp;
        GridGenerator::subdivided_hyper_rectangle(tria_temp, subdivisions, p1, p2);
        GridGenerator::convert_hypercube_to_simplex_mesh(tria_temp, tria_serial);
      }
    else
      {
        GridGenerator::subdivided_hyper_rectangle(tria_serial, subdivisions, p1, p2);
      }

    GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), tria_serial);

    tria.create_triangulation(
      TriangulationDescription::Utilities::create_description_from_triangulation(tria_serial,
                                                                                 MPI_COMM_WORLD));
  }

  std::unique_ptr<Mapping<dim, spacedim>>       mapping;
  std::unique_ptr<FiniteElement<dim, spacedim>> fe;
  std::unique_ptr<Quadrature<dim>>              quadrature;

  if (do_simplex)
    {
      mapping    = std::make_unique<MappingFE<dim, spacedim>>(Simplex::FE_P<dim, spacedim>(1));
      fe         = std::make_unique<Simplex::FE_P<dim, spacedim>>(degree);
      quadrature = std::make_unique<Simplex::QGauss<dim>>(degree + 1);
    }
  else
    {
      mapping    = std::make_unique<MappingQGeneric<dim, spacedim>>(1);
      fe         = std::make_unique<FE_Q<dim, spacedim>>(degree);
      quadrature = std::make_unique<QGauss<dim>>(degree + 1);
    }

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(*fe);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
  constraints.close();

  LinearAlgebra::distributed::Vector<double> x, b;

  PoissonOperator poisson_operator(*mapping, dof_handler, constraints, *quadrature, x, b);

  {
    const auto start = std::chrono::system_clock::now();

    for (unsigned int i = 0; i < 100; ++i)
      poisson_operator.vmult(x, b);

    const auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - start).count() /
      1e9;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << tria.n_global_active_cells() << " " << dof_handler.n_dofs() << " " << duration
                << std::endl;
  }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim            = argc <= 1 ? 3 : atoi(argv[1]);
  const unsigned int degree         = argc <= 2 ? 2 : atoi(argv[2]);
  const bool         do_simplex     = argc <= 3 ? 1 : atoi(argv[3]);
  const bool         do_matrix_free = argc <= 3 ? 1 : atoi(argv[4]);
  const unsigned int s_max          = argc <= 3 ? 9 : atoi(argv[5]);

  std::vector<unsigned int> subdivisions(dim, 1);

  for (unsigned int s = 0, c = dim; s < s_max; ++s, c += 1)
    {
      if (dim == 2 && do_matrix_free == false)
        test<PoissonOperatorMatrixBased<2>>(degree, do_simplex, subdivisions);

      if (dim == 2 && do_matrix_free == true)
        test<PoissonOperatorMatrixFree<2>>(degree, do_simplex, subdivisions);

      if (dim == 3 && do_matrix_free == false)
        test<PoissonOperatorMatrixBased<3>>(degree, do_simplex, subdivisions);

      if (dim == 3 && do_matrix_free == true)
        test<PoissonOperatorMatrixFree<3>>(degree, do_simplex, subdivisions);

      subdivisions[s % dim] *= 2;
    }
}
