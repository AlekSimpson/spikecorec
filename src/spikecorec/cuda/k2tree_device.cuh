#pragma once

// CUDA port of the k^2-tree bit-walk helpers in metal/k2tree_device.metalinc, for the precompiled
// CUDA kernels (kernels.cu, weight_fit_kernels.cu). The runtime-compiled master kernel carries its
// own copy in kernel_boilerplate.cpp, since NVRTC compiles it with no include path.

// Bit positions are unsigned, so a tree may hold up to 2^32 bits.
__device__ inline unsigned int k2t_get_bit(const unsigned int *words, unsigned int bit_index) {
    return (words[bit_index >> 5] >> (bit_index & 31u)) & 1u;
}

__device__ inline unsigned int k2t_rank1_exclusive(
    const unsigned int *internal_words,
    const unsigned int *superblock_table,
    const unsigned short *subblock_table,
    unsigned int position,
    int superblock_size
) {
    const unsigned int word_index = position >> 5;
    const unsigned int bit_offset = position & 31u;
    const unsigned int superblock_index = word_index / static_cast<unsigned int>(superblock_size);
    const unsigned int superblock_base = superblock_table[superblock_index];
    const unsigned int subblock_base = static_cast<unsigned int>(subblock_table[word_index]);
    const unsigned int partial_word_mask = (bit_offset == 0u) ? 0u : ((1u << bit_offset) - 1u);
    const unsigned int partial_word_popcount = __popc(internal_words[word_index] & partial_word_mask);
    return superblock_base + subblock_base + partial_word_popcount;
}

#define MAX_K2TREE_HEIGHT 32

// Node indices are unsigned, so a walk that finds no node returns this (UINT_MAX) instead of -1.
#define K2T_NO_NEIGHBOR 4294967295u

// Returns row source_node's target_slot-th target (0-indexed, ascending), or K2T_NO_NEIGHBOR when
// it has fewer. Walks only populated subtrees on the row, by an iterative DFS with an explicit
// stack; depth is tree_height, at most MAX_K2TREE_HEIGHT.
__device__ inline unsigned int k2t_find_nth_neighbor(
    const unsigned int *internal_node_words,
    const unsigned int *leaf_node_words,
    const unsigned int *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    int branching_factor,
    int superblock_size_words,
    int node_count,
    int padded_node_count,
    int tree_height,
    int internal_bit_count,
    unsigned int source_node,
    int target_slot
) {
    if (tree_height == 0 || source_node >= static_cast<unsigned int>(node_count) || target_slot < 0) {
        return K2T_NO_NEIGHBOR;
    }

    const int branching_factor_squared = branching_factor * branching_factor;

    unsigned int stack_row_base[MAX_K2TREE_HEIGHT];
    unsigned int stack_column_base[MAX_K2TREE_HEIGHT];
    int stack_block_size[MAX_K2TREE_HEIGHT];
    unsigned int stack_bit_offset[MAX_K2TREE_HEIGHT];
    int stack_next_column[MAX_K2TREE_HEIGHT];

    stack_row_base[0] = 0;
    stack_column_base[0] = 0;
    stack_block_size[0] = padded_node_count;
    stack_bit_offset[0] = 0;
    stack_next_column[0] = 0;

    int stack_top = 0;
    int neighbors_seen = 0;

    while (stack_top >= 0) {
        const int level = stack_top;
        const int column_offset = stack_next_column[level];
        if (column_offset >= branching_factor) {
            stack_top -= 1;
            continue;
        }
        stack_next_column[level] = column_offset + 1;

        const unsigned int row_base = stack_row_base[level];
        const unsigned int column_base = stack_column_base[level];
        const int block_size = stack_block_size[level];
        const unsigned int level_bit_offset = stack_bit_offset[level];

        const int child_block_size = block_size / branching_factor;
        const int row_offset = static_cast<int>((source_node - row_base) / static_cast<unsigned int>(child_block_size));
        const int child_flat_index = row_offset * branching_factor + column_offset;
        const unsigned int bit_position = level_bit_offset + static_cast<unsigned int>(child_flat_index);

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                const unsigned int target_node = column_base + static_cast<unsigned int>(column_offset);
                if (target_node < static_cast<unsigned int>(node_count)) {
                    if (neighbors_seen == target_slot) {
                        return target_node;
                    }
                    neighbors_seen += 1;
                }
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            const unsigned int rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                                    rank_subblock_table, bit_position, superblock_size_words) + 1u;
            const int child_level = stack_top + 1;
            const unsigned int raw_offset = static_cast<unsigned int>(branching_factor_squared) * rank_inclusive;
            stack_row_base[child_level] = row_base + static_cast<unsigned int>(row_offset * child_block_size);
            stack_column_base[child_level] = column_base + static_cast<unsigned int>(column_offset * child_block_size);
            stack_block_size[child_level] = child_block_size;
            stack_bit_offset[child_level] = (child_level == tree_height - 1)
                ? (raw_offset - static_cast<unsigned int>(internal_bit_count))
                : raw_offset;
            stack_next_column[child_level] = 0;
            stack_top = child_level;
        }
    }

    return K2T_NO_NEIGHBOR;
}

// Returns row source_node's next target, resuming the DFS from the caller's stack, or
// K2T_NO_NEIGHBOR when done. The caller owns the stack arrays (MAX_K2TREE_HEIGHT each) and
// initializes them once: stack_row_base[0] = 0, stack_column_base[0] = 0,
// stack_block_size[0] = padded_node_count, stack_bit_offset[0] = 0, stack_next_column[0] = 0,
// stack_top = 0 (or -1 to short-circuit).
__device__ inline unsigned int k2t_next_neighbor(
    const unsigned int *internal_node_words,
    const unsigned int *leaf_node_words,
    const unsigned int *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    int branching_factor,
    int superblock_size_words,
    int node_count,
    int tree_height,
    int internal_bit_count,
    unsigned int source_node,
    unsigned int *stack_row_base,
    unsigned int *stack_column_base,
    int *stack_block_size,
    unsigned int *stack_bit_offset,
    int *stack_next_column,
    int &stack_top
) {
    const int branching_factor_squared = branching_factor * branching_factor;

    while (stack_top >= 0) {
        const int level = stack_top;
        const int column_offset = stack_next_column[level];
        if (column_offset >= branching_factor) {
            stack_top -= 1;
            continue;
        }
        stack_next_column[level] = column_offset + 1;

        const unsigned int row_base = stack_row_base[level];
        const unsigned int column_base = stack_column_base[level];
        const int block_size = stack_block_size[level];
        const unsigned int level_bit_offset = stack_bit_offset[level];

        const int child_block_size = block_size / branching_factor;
        const int row_offset = static_cast<int>((source_node - row_base) / static_cast<unsigned int>(child_block_size));
        const int child_flat_index = row_offset * branching_factor + column_offset;
        const unsigned int bit_position = level_bit_offset + static_cast<unsigned int>(child_flat_index);

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                const unsigned int target_node = column_base + static_cast<unsigned int>(column_offset);
                if (target_node < static_cast<unsigned int>(node_count)) {
                    return target_node;
                }
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            const unsigned int rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                                    rank_subblock_table, bit_position, superblock_size_words) + 1u;
            const int child_level = stack_top + 1;
            const unsigned int raw_offset = static_cast<unsigned int>(branching_factor_squared) * rank_inclusive;
            stack_row_base[child_level] = row_base + static_cast<unsigned int>(row_offset * child_block_size);
            stack_column_base[child_level] = column_base + static_cast<unsigned int>(column_offset * child_block_size);
            stack_block_size[child_level] = child_block_size;
            stack_bit_offset[child_level] = (child_level == tree_height - 1)
                ? (raw_offset - static_cast<unsigned int>(internal_bit_count))
                : raw_offset;
            stack_next_column[child_level] = 0;
            stack_top = child_level;
        }
    }
    return K2T_NO_NEIGHBOR;
}

// Returns column target_node's next source, ascending, resuming the DFS from the caller's stack,
// or K2T_NO_NEIGHBOR when done: the column mirror of k2t_next_neighbor, which fixes the column
// offset at each level and steps through the row offsets. The caller initializes the stack the
// same way, with stack_next_row in place of stack_next_column.
__device__ inline unsigned int k2t_next_predecessor(
    const unsigned int *internal_node_words,
    const unsigned int *leaf_node_words,
    const unsigned int *rank_superblock_table,
    const unsigned short *rank_subblock_table,
    int branching_factor,
    int superblock_size_words,
    int node_count,
    int tree_height,
    int internal_bit_count,
    unsigned int target_node,
    unsigned int *stack_row_base,
    unsigned int *stack_column_base,
    int *stack_block_size,
    unsigned int *stack_bit_offset,
    int *stack_next_row,
    int &stack_top
) {
    const int branching_factor_squared = branching_factor * branching_factor;

    while (stack_top >= 0) {
        const int level = stack_top;
        const int row_offset = stack_next_row[level];
        if (row_offset >= branching_factor) {
            stack_top -= 1;
            continue;
        }
        stack_next_row[level] = row_offset + 1;

        const unsigned int row_base = stack_row_base[level];
        const unsigned int column_base = stack_column_base[level];
        const int block_size = stack_block_size[level];
        const unsigned int level_bit_offset = stack_bit_offset[level];

        const int child_block_size = block_size / branching_factor;
        const int column_offset = static_cast<int>((target_node - column_base) / static_cast<unsigned int>(child_block_size));
        const int child_flat_index = row_offset * branching_factor + column_offset;
        const unsigned int bit_position = level_bit_offset + static_cast<unsigned int>(child_flat_index);

        if (level == tree_height - 1) {
            if (k2t_get_bit(leaf_node_words, bit_position)) {
                const unsigned int source_node = row_base + static_cast<unsigned int>(row_offset);
                if (source_node < static_cast<unsigned int>(node_count)) {
                    return source_node;
                }
            }
        } else if (k2t_get_bit(internal_node_words, bit_position)) {
            const unsigned int rank_inclusive = k2t_rank1_exclusive(internal_node_words, rank_superblock_table,
                                                                    rank_subblock_table, bit_position, superblock_size_words) + 1u;
            const int child_level = stack_top + 1;
            const unsigned int raw_offset = static_cast<unsigned int>(branching_factor_squared) * rank_inclusive;
            stack_row_base[child_level] = row_base + static_cast<unsigned int>(row_offset * child_block_size);
            stack_column_base[child_level] = column_base + static_cast<unsigned int>(column_offset * child_block_size);
            stack_block_size[child_level] = child_block_size;
            stack_bit_offset[child_level] = (child_level == tree_height - 1)
                ? (raw_offset - static_cast<unsigned int>(internal_bit_count))
                : raw_offset;
            stack_next_row[child_level] = 0;
            stack_top = child_level;
        }
    }
    return K2T_NO_NEIGHBOR;
}
