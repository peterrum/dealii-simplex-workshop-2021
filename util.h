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

// Common utility functions for the examples.

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

// Return the communicator of a MeshType (Triangulation or DoFHandler).
template <typename MeshType>
MPI_Comm
get_mpi_comm(const MeshType &mesh)
{
  const auto *tria_parallel =
    dynamic_cast<const parallel::TriangulationBase<MeshType::dimension, MeshType::space_dimension> *>(
      &(mesh.get_triangulation()));

  return tria_parallel != nullptr ? tria_parallel->get_communicator() : MPI_COMM_SELF;
}



// (Sparse)-matrix-based Laplace operator. 
template <int dim_in, int spacedim_in = dim_in>
class PoissonOperatorMatrixBased
{
public:
  static const int dim      = dim_in;
  static const int spacedim = spacedim_in;

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  // Normal (non-hp) case.
  PoissonOperatorMatrixBased(const Mapping<dim, spacedim> &   mapping,
                             const DoFHandler<dim, spacedim> &dof_handler,
                             const AffineConstraints<double> &constraints,
                             const Quadrature<dim> &          quad,
                             VectorType &                     x,
                             VectorType &                     system_rhs)
  {
    TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(), get_mpi_comm(dof_handler));
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dsp.compress();

    system_matrix.reinit(dsp);

    const auto partitioner = create_partitioner(dof_handler);

    x.reinit(partitioner);
    system_rhs.reinit(partitioner);

    // Assemble system matrix and right-hand-side vector.
    FEValues<dim, spacedim> fe_values(mapping,
                                      dof_handler.get_fe(),
                                      quad,
                                      update_values | update_gradients |
                                        update_JxW_values);

    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        fe_values.reinit(cell);

        const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_matrix = 0;
        cell_rhs.reinit(dofs_per_cell);
        cell_rhs = 0;

        for (const auto q : fe_values.quadrature_point_indices())
          {
            for (const auto i : fe_values.dof_indices())
              for (const auto j : fe_values.dof_indices())
                cell_matrix(i, j) += (fe_values.shape_grad(i, q) * //
                                      fe_values.shape_grad(j, q) * //
                                      fe_values.JxW(q));           //

            for (const unsigned int i : fe_values.dof_indices())
              cell_rhs(i) += (fe_values.shape_value(i, q) * //
                              1. *                          //
                              fe_values.JxW(q));            //
          }

        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }

    system_rhs.compress(VectorOperation::values::add);
    system_matrix.compress(VectorOperation::values::add);
  }

  // hp-case.
  PoissonOperatorMatrixBased(const hp::MappingCollection<dim, spacedim> &mapping,
                             const DoFHandler<dim, spacedim> &           dof_handler,
                             const AffineConstraints<double> &           constraints,
                             const hp::QCollection<dim> &                quad,
                             VectorType &                                x,
                             VectorType &                                system_rhs)
  {
    TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(), get_mpi_comm(dof_handler));
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dsp.compress();

    system_matrix.reinit(dsp);

    const auto partitioner = create_partitioner(dof_handler);

    x.reinit(partitioner);
    system_rhs.reinit(partitioner);

    // Assemble system matrix and right-hand-side vector.
    hp::FEValues<dim, spacedim> hp_fe_values(mapping,
                                             dof_handler.get_fe_collection(),
                                             quad,
                                             update_values | update_gradients |
                                               update_JxW_values);

    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        hp_fe_values.reinit(cell);

        auto &fe_values = hp_fe_values.get_present_fe_values();

        const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_matrix = 0;
        cell_rhs.reinit(dofs_per_cell);
        cell_rhs = 0;

        for (const auto q : fe_values.quadrature_point_indices())
          {
            for (const auto i : fe_values.dof_indices())
              for (const auto j : fe_values.dof_indices())
                cell_matrix(i, j) += (fe_values.shape_grad(i, q) * //
                                      fe_values.shape_grad(j, q) * //
                                      fe_values.JxW(q));           //

            for (const unsigned int i : fe_values.dof_indices())
              cell_rhs(i) += (fe_values.shape_value(i, q) * //
                              1. *                          //
                              fe_values.JxW(q));            //
          }

        local_dof_indices.resize(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }

    system_rhs.compress(VectorOperation::values::add);
    system_matrix.compress(VectorOperation::values::add);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    system_matrix.vmult(dst, src);
  }

private:
  std::shared_ptr<const Utilities::MPI::Partitioner>
  create_partitioner(const DoFHandler<dim, spacedim> &dof_handler)
  {
    IndexSet locally_relevant_dofs;

    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    return std::make_shared<const Utilities::MPI::Partitioner>(dof_handler.locally_owned_dofs(),
                                                               locally_relevant_dofs,
                                                               get_mpi_comm(dof_handler));
  }


  TrilinosWrappers::SparseMatrix system_matrix;
};



// Matrix-free Laplace operator.
template <int dim_in, int spacedim_in = dim_in>
class PoissonOperatorMatrixFree
{
public:
  static const int dim      = dim_in;
  static const int spacedim = spacedim_in;

  using VectorType     = LinearAlgebra::distributed::Vector<double>;
  using FECellInegrals = FEEvaluation<dim, -1, 0, 1, double>;

  template <typename MappingType, typename QuadratureType>
  PoissonOperatorMatrixFree(const MappingType &              mapping,
                            const DoFHandler<dim, spacedim> &dof_handler,
                            const AffineConstraints<double> &constraints,
                            const QuadratureType &           quad,
                            VectorType &                     x,
                            VectorType &                     b)
  {
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_gradients | update_values;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, additional_data);

    matrix_free.initialize_dof_vector(x);
    matrix_free.initialize_dof_vector(b);

    const int dummy = 0;

    matrix_free.template cell_loop<VectorType, int>(
      [](const auto &matrix_free, auto &dst, const auto &, const auto cells) {
        FECellInegrals phi(matrix_free, cells);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_value(1.0, q);

            phi.integrate_scatter(EvaluationFlags::values, dst);
          }
      },
      b,
      dummy,
      true);
  }

  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.template cell_loop<VectorType, VectorType>(
      [](const auto &matrix_free, auto &dst, const auto &src, const auto cells) {
        FECellInegrals phi(matrix_free, cells);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, EvaluationFlags::gradients);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_gradient(phi.get_gradient(q), q);

            phi.integrate_scatter(EvaluationFlags::gradients, dst);
          }
      },
      dst,
      src,
      true);
  }

private:
  MatrixFree<dim, double> matrix_free;
};
