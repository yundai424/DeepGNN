// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "src/cc/lib/graph/graph.h"
#include "src/cc/lib/graph/partition.h"
#include "src/cc/lib/graph/sampler.h"
#include "src/cc/lib/graph/xoroshiro.h"
#include "src/cc/tests/mocks.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <span>
#include <vector>

#include "boost/random/uniform_int_distribution.hpp"
#include "gtest/gtest.h"

class TemporalTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        {
            TestGraph::MemoryGraph m1;
            m1.m_nodes.push_back(
                TestGraph::Node{.m_id = 0,
                                .m_type = 0,
                                .m_weight = 1.0f,
                                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 2.0f}}}});

            m1.m_nodes.push_back(TestGraph::Node{
                .m_id = 1,
                .m_type = 1,
                .m_weight = 1.0f,
                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 7.0f}}}});

            m1.m_watermark = 1;
            using ts_pair = std::pair<snark::Timestamp, snark::Timestamp>;
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{1, 2});
            m1.m_edge_timestamps.emplace_back(ts_pair{2, 3});

            // Initialize graph
            auto path = std::filesystem::temp_directory_path();
            TestGraph::convert(path, "0_0", std::move(m1), 2);
            snark::Metadata metadata(path.string());
            m_single_partition_graph =
                std::make_unique<snark::Graph>(std::move(metadata), std::vector<std::string>{path.string()},
                                               std::vector<uint32_t>{0}, snark::PartitionStorageType::memory);
        }
        {
            TestGraph::MemoryGraph m1;
            m1.m_nodes.push_back(
                TestGraph::Node{.m_id = 0,
                                .m_type = 0,
                                .m_weight = 1.0f,
                                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{1, 0, 1.0f}, {2, 0, 1.0f}}}});

            m1.m_nodes.push_back(TestGraph::Node{
                .m_id = 1,
                .m_type = 1,
                .m_weight = 1.0f,
                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{3, 0, 1.0f}, {4, 0, 1.0f}, {5, 1, 1.0f}}}});

            m1.m_watermark = 2;
            using ts_pair = std::pair<snark::Timestamp, snark::Timestamp>;
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m1.m_edge_timestamps.emplace_back(ts_pair{1, -1});
            m1.m_edge_timestamps.emplace_back(ts_pair{1, -1});
            TestGraph::MemoryGraph m2;
            m2.m_nodes.push_back(
                TestGraph::Node{.m_id = 1,
                                .m_type = 1,
                                .m_neighbors{std::vector<TestGraph::NeighborRecord>{{6, 1, 1.5f}, {7, 1, 3.0f}}}});

            m2.m_watermark = 3;
            m2.m_edge_timestamps.emplace_back(ts_pair{0, 1});
            m2.m_edge_timestamps.emplace_back(ts_pair{2, 3});

            // Initialize Graph
            auto path = std::filesystem::temp_directory_path();
            TestGraph::convert(path, "0_0", std::move(m1), 2);
            TestGraph::convert(path, "1_0", std::move(m2), 2);
            snark::Metadata metadata(path.string());
            m_multi_partition_graph = std::make_unique<snark::Graph>(
                std::move(metadata), std::vector<std::string>{path.string(), path.string()},
                std::vector<uint32_t>{0, 1}, snark::PartitionStorageType::memory);
        }
    }

    std::unique_ptr<snark::Graph> m_single_partition_graph;
    std::unique_ptr<snark::Graph> m_multi_partition_graph;
};

// Neighbor Count Tests
TEST_F(TemporalTest, GetNeigborCountSinglePartition)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {0};
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::Timestamp> ts = {2, 2};

    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    ts = {0, 0};
    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({2, 1}), output_neighbors_count);

    // Check for different singe edge type filter
    types = {1};
    ts = {2, 2};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);

    // Check for both edge types
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);

    // Check returns 0 for unsatisfying edge types
    types = {-1, 100};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    // Invalid node ids
    nodes = {99, 100};
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);
}

TEST_F(TemporalTest, GetFullNeighborSinglePartition)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {0};
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::Timestamp> ts = {2, 2};

    std::vector<snark::NodeId> output_neighbor_ids;
    std::vector<snark::Type> output_neighbor_types;
    std::vector<float> output_neighbors_weights;
    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights,
                                           std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    ts = {0, 0};
    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights,
                                           std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({1, 2, 4}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({0, 0, 0}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({1.f, 2.f, 1.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<uint64_t>({2, 1}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check for different singe edge type filter
    types = {1};
    ts = {2, 2};
    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights,
                                           std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({5}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({1}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({7.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check for both edge types
    types = {0, 1};
    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights,
                                           std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({5}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({1}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({7.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check returns 0 for unsatisfying edge types
    types = {-1, 100};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights,
                                           std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    // Invalid node ids
    nodes = {99, 100};
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_single_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                           output_neighbor_types, output_neighbors_weights,
                                           std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);
}

TEST_F(TemporalTest, GetNeigborCountMultiplePartitions)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {1};
    std::vector<uint64_t> output_neighbors_count(nodes.size());

    m_multi_partition_graph->NeighborCount(std::span(nodes), std::span(types), {}, output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 3}), output_neighbors_count);

    // Check for multiple edge types
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    std::vector<snark::Timestamp> ts = {2, 2};
    m_multi_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 2}), output_neighbors_count);

    // Check non-existent edge types functionality
    types = {-1, 100};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_multi_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    // Check invalid node ids handling
    nodes = {99, 100};
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_multi_partition_graph->NeighborCount(std::span(nodes), std::span(types), std::span(ts), output_neighbors_count);
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);
}

TEST_F(TemporalTest, GetFullNeighborMultiplePartitions)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {1};
    std::vector<uint64_t> output_neighbors_count(nodes.size());
    std::vector<snark::Timestamp> ts = {2, 2};

    std::vector<snark::NodeId> output_neighbor_ids;
    std::vector<snark::Type> output_neighbor_types;
    std::vector<float> output_neighbors_weights;
    m_multi_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                          output_neighbor_types, output_neighbors_weights,
                                          std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({5}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({1}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({1.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check for multiple edge types
    types = {0, 1};
    m_multi_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                          output_neighbor_types, output_neighbors_weights,
                                          std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({3, 5}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({0, 1}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({1.f, 1.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<uint64_t>({0, 2}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check for different singe edge type filter
    types = {1};
    ts = {2, 2};
    m_multi_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                          output_neighbor_types, output_neighbors_weights,
                                          std::span(output_neighbors_count));
    EXPECT_EQ(std::vector<snark::NodeId>({5}), output_neighbor_ids);
    output_neighbor_ids.clear();
    EXPECT_EQ(std::vector<snark::Type>({1}), output_neighbor_types);
    output_neighbor_types.clear();
    EXPECT_EQ(std::vector<float>({1.f}), output_neighbors_weights);
    output_neighbors_weights.clear();
    EXPECT_EQ(std::vector<uint64_t>({0, 1}), output_neighbors_count);
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    // Check returns 0 for unsatisfying edge types
    types = {-1, 100};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_multi_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                          output_neighbor_types, output_neighbors_weights,
                                          std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);

    // Invalid node ids
    nodes = {99, 100};
    types = {0, 1};
    std::fill_n(output_neighbors_count.begin(), 2, -1);

    m_multi_partition_graph->FullNeighbor(std::span(nodes), std::span(types), std::span(ts), output_neighbor_ids,
                                          output_neighbor_types, output_neighbors_weights,
                                          std::span(output_neighbors_count));
    EXPECT_TRUE(output_neighbor_ids.empty());
    EXPECT_TRUE(output_neighbor_types.empty());
    EXPECT_TRUE(output_neighbors_weights.empty());
    EXPECT_EQ(std::vector<uint64_t>({0, 0}), output_neighbors_count);
}

TEST_F(TemporalTest, GetSampleNeighborsMultiplePartitions)
{
    // Check for singe edge type filter
    std::vector<snark::NodeId> nodes = {0, 1};
    std::vector<snark::Type> types = {1};
    std::vector<snark::Timestamp> ts = {2, 2};
    size_t sample_count = 2;

    std::vector<snark::NodeId> output_neighbor_ids(sample_count * nodes.size());
    std::vector<snark::Type> output_neighbor_types(sample_count * nodes.size());
    std::vector<float> output_neighbors_weights(sample_count * nodes.size());
    std::vector<float> output_neighbors_total_weights(nodes.size());
    m_multi_partition_graph->SampleNeighbor(33, std::span(nodes), std::span(types), std::span(ts), sample_count,
                                            std::span(output_neighbor_ids), std::span(output_neighbor_types),
                                            std::span(output_neighbors_weights),
                                            std::span(output_neighbors_total_weights), 42, 0.5f, 13);
    // Only available nodes based on time/type are 5 and 7
    EXPECT_EQ(std::vector<snark::NodeId>({5, 5, 7, 7}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({1, 1, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f}), output_neighbors_weights);
    std::fill(std::begin(output_neighbors_weights), std::end(output_neighbors_weights), -1);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f}), output_neighbors_total_weights);
    std::fill(std::begin(output_neighbors_total_weights), std::end(output_neighbors_total_weights), -1);

    // Check for multiple edge types
    types = {0, 1};
    m_multi_partition_graph->SampleNeighbor(33, std::span(nodes), std::span(types), std::span(ts), sample_count,
                                            std::span(output_neighbor_ids), std::span(output_neighbor_types),
                                            std::span(output_neighbors_weights),
                                            std::span(output_neighbors_total_weights), 42, 0.5f, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({5, 3, 4, 1}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({1, 1, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f}), output_neighbors_weights);
    std::fill(std::begin(output_neighbors_weights), std::end(output_neighbors_weights), -1);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f}), output_neighbors_total_weights);
    std::fill(std::begin(output_neighbors_total_weights), std::end(output_neighbors_total_weights), -1);

    // Check for different singe edge type filter
    types = {1};
    ts = {2, 2};
    m_multi_partition_graph->SampleNeighbor(33, std::span(nodes), std::span(types), std::span(ts), sample_count,
                                            std::span(output_neighbor_ids), std::span(output_neighbor_types),
                                            std::span(output_neighbors_weights),
                                            std::span(output_neighbors_total_weights), 42, 0.5f, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({5, 3, 4, 1}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({1, 1, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f}), output_neighbors_weights);
    std::fill(std::begin(output_neighbors_weights), std::end(output_neighbors_weights), -1);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f}), output_neighbors_total_weights);
    std::fill(std::begin(output_neighbors_total_weights), std::end(output_neighbors_total_weights), -1);

    // Check returns 0 for unsatisfying edge types
    types = {-1, 100};

    m_multi_partition_graph->SampleNeighbor(33, std::span(nodes), std::span(types), std::span(ts), sample_count,
                                            std::span(output_neighbor_ids), std::span(output_neighbor_types),
                                            std::span(output_neighbors_weights),
                                            std::span(output_neighbors_total_weights), 42, 0.5f, 13);
    EXPECT_EQ(std::vector<snark::NodeId>({5, 3, 4, 1}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({1, 1, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f}), output_neighbors_weights);
    std::fill(std::begin(output_neighbors_weights), std::end(output_neighbors_weights), -1);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f}), output_neighbors_total_weights);
    std::fill(std::begin(output_neighbors_total_weights), std::end(output_neighbors_total_weights), -1);

    // Invalid node ids
    nodes = {99, 100};
    types = {0, 1};

    m_multi_partition_graph->SampleNeighbor(33, std::span(nodes), std::span(types), std::span(ts), sample_count,
                                            std::span(output_neighbor_ids), std::span(output_neighbor_types),
                                            std::span(output_neighbors_weights),
                                            std::span(output_neighbors_total_weights), 42, 0.5f, 13);

    EXPECT_EQ(std::vector<snark::NodeId>({5, 3, 4, 1}), output_neighbor_ids);
    std::fill(std::begin(output_neighbor_ids), std::end(output_neighbor_ids), -1);
    EXPECT_EQ(std::vector<snark::Type>({1, 1, 1, 1}), output_neighbor_types);
    std::fill(std::begin(output_neighbor_types), std::end(output_neighbor_types), -1);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f}), output_neighbors_weights);
    std::fill(std::begin(output_neighbors_weights), std::end(output_neighbors_weights), -1);
    EXPECT_EQ(std::vector<float>({1.f, 1.f, 1.f, 1.f}), output_neighbors_total_weights);
    std::fill(std::begin(output_neighbors_total_weights), std::end(output_neighbors_total_weights), -1);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
