import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import time
from tqdm import tqdm
import json
import random
import string
from pymilvus import MilvusClient, DataType
import matplotlib.pyplot as plt
import struct
from openai import OpenAI
import faker
import os
from dotenv import load_dotenv
import concurrent.futures
import threading
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Initialize Faker for generating fake text
fake = faker.Faker()

class VectorBenchmark:
    def __init__(self, host='localhost', port='19530'):
        self.host = host
        self.port = port
        
        # Initialize MilvusClient instead of using connections
        self.client = MilvusClient(
            uri=f"http://{host}:{port}",
            token="root:Milvus"  # Use appropriate credentials
        )
        
        # CPU index configurations
        self.index_configs = {
            'IVF_FLAT': {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            },
            'IVF_SQ8': {
                "metric_type": "L2",
                "index_type": "IVF_SQ8",
                "params": {"nlist": 1024}
            },
            'HNSW': {
                "metric_type": "L2",
                "index_type": "HNSW",
                "params": {"M": 48, "efConstruction": 500}
            }
        }
        
        # GPU index configurations
        self.gpu_index_configs = {
            'GPU_CAGRA': {
                "metric_type": "L2",
                "index_type": "GPU_CAGRA",
                "params": {
                    "intermediate_graph_degree": 128,
                    "graph_degree": 64,
                    "build_algo": "IVF_PQ",
                    "cache_dataset_on_device": "false",
                    "adapt_for_cpu": "false"
                }
            },
            'GPU_IVF_FLAT': {
                "metric_type": "L2",
                "index_type": "GPU_IVF_FLAT",
                "params": {
                    "nlist": 1024,
                    "cache_dataset_on_device": "false"
                }
            },
            'GPU_IVF_PQ': {
                "metric_type": "L2",
                "index_type": "GPU_IVF_PQ",
                "params": {
                    "nlist": 1024,
                    "m": 0,  # Will be set based on dimension
                    "nbits": 8,
                    "cache_dataset_on_device": "false"
                }
            },
            'GPU_BRUTE_FORCE': {
                "metric_type": "L2",
                "index_type": "GPU_BRUTE_FORCE",
                "params": {}
            }
        }

    def get_embedding(self, texts):
        """Generate embeddings using OpenAI API"""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        for text in texts:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings)

    def generate_random_text(self, length=100):
        return ''.join(random.choices(string.ascii_letters + ' ', k=length))

    def create_collection(self, dim=1536, collection_name="benchmark_collection", 
                         index_type="IVF_FLAT", create_index=True, index_config=None):
        # Drop collection if it exists
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        
        # Create schema
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        
        # Add fields to schema
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dim)
        
        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema
        )
        
        # Create index if requested
        index_time = 0
        if create_index:
            start_time = time.time()
            
            # Prepare index parameters
            index_params = self.client.prepare_index_params()
            
            # Use provided index config or default
            if index_config is None:
                if index_type in self.index_configs:
                    index_config = self.index_configs[index_type]
                elif index_type in self.gpu_index_configs:
                    index_config = self.gpu_index_configs[index_type]
                else:
                    raise ValueError(f"Unknown index type: {index_type}")
            
            # Add index for vector field
            index_params.add_index(
                field_name="embedding",
                index_type=index_config["index_type"],
                metric_type=index_config["metric_type"],
                params=index_config["params"]
            )
            
            # Create index
            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
            
            index_time = time.time() - start_time
        
        return collection_name, index_time

    def generate_embeddings_file(self, num_vectors=1000000, batch_size=100, output_file="embeddings.bin"):
        with open(output_file, 'wb') as f:
            f.write(struct.pack('ii', 1, 384))  # Version 1, 384 dimensions
           
            for i in tqdm(range(0, num_vectors, batch_size)):
                batch_size = min(batch_size, num_vectors - i)
                texts = [self.generate_random_text() for _ in range(batch_size)]
                embeddings = self.get_embedding(texts)
               
                for text, embedding in zip(texts, embeddings):
                    encoded_text = text.encode('utf-8')
                    f.write(struct.pack('i', len(encoded_text)))
                    f.write(encoded_text)
                    f.write(embedding.astype(np.float32).tobytes())
       
        return output_file

    def load_embeddings(self, filename):
        embeddings = []
        texts = []
        with open(filename, 'rb') as f:
            version, dims = struct.unpack('ii', f.read(8))
            while True:
                try:
                    text_len = struct.unpack('i', f.read(4))[0]
                    text = f.read(text_len).decode('utf-8')
                    embedding = np.frombuffer(f.read(dims * 4), dtype=np.float32)
                    texts.append(text)
                    embeddings.append(embedding)
                except:
                    break
        return texts, np.array(embeddings)

    def benchmark_insertion(self, collection_name, batch_sizes=[1000, 5000, 10000],
                          total_vectors=1000000, dim=1536, index_type="IVF_FLAT"):
        results = []
       
        # Get the appropriate index configuration
        if index_type in self.index_configs:
            index_config = self.index_configs[index_type]
        elif index_type in self.gpu_index_configs:
            index_config = self.gpu_index_configs[index_type]
        else:
            raise ValueError(f"Unknown index type: {index_type}")
       
        for batch_size in batch_sizes:
            # Generate random vectors instead of using embeddings
            texts = []
            embeddings = []
            for i in range(0, total_vectors, batch_size):
                current_batch_size = min(batch_size, total_vectors - i)
                
                # Generate random texts and vectors
                batch_texts = [self.generate_random_text() for _ in range(current_batch_size)]
                batch_embeddings = np.random.rand(current_batch_size, dim).astype(np.float32)
                
                texts.extend(batch_texts)
                embeddings.extend(batch_embeddings)
            
            # Insert data
            insert_start_time = time.time()
            
            # Prepare data for insertion
            data = []
            for i in range(len(texts)):
                data.append({
                    "id": i,
                    "text": texts[i],
                    "embedding": embeddings[i].tolist()
                })
            
            # Insert in batches
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i+batch_size]
                self.client.insert(
                    collection_name=collection_name,
                    data=batch_data
                )
            
            insert_time = time.time() - insert_start_time
            
            # Create index after insertion
            index_start_time = time.time()
            
            # Prepare index parameters
            index_params = self.client.prepare_index_params()
            
            # Add index for vector field
            index_params.add_index(
                field_name="embedding",
                index_type=index_config["index_type"],
                metric_type=index_config["metric_type"],
                params=index_config["params"]
            )
            
            # Create index
            self.client.create_index(
                collection_name=collection_name,
                index_params=index_params
            )
            
            index_time_after = time.time() - index_start_time
           
            results.append({
                'batch_size': batch_size,
                'insert_time': insert_time,
                'vectors_per_second': total_vectors / insert_time,
                'index_time_after_insert': index_time_after
            })
           
        return results

    def benchmark_search(self, collection_name, target_embedding, target_id,
                        num_searches=100, topk=100):
        # Load collection
        self.client.load_collection(collection_name)
        
        # Get the index type from collection
        index_info = self.client.describe_index(
            collection_name=collection_name,
            index_name="embedding"  # Index name is the same as field name by default
        )
        index_type = index_info['index_type']
        
        # Define search parameters for different index types
        search_params = {
            # CPU indexes
            'IVF_FLAT': [
                {"nprobe": 10},
                {"nprobe": 50},
                {"nprobe": 100},
                {"nprobe": 500}
            ],
            'IVF_SQ8': [
                {"nprobe": 10},
                {"nprobe": 50},
                {"nprobe": 100},
                {"nprobe": 500}
            ],
            'HNSW': [
                {"ef": 10},
                {"ef": 50},
                {"ef": 100},
                {"ef": 500}
            ],
            # GPU indexes
            'GPU_CAGRA': [
                {"itopk_size": 64, "search_width": 1},
                {"itopk_size": 128, "search_width": 2},
                {"itopk_size": 256, "search_width": 4},
                {"itopk_size": 512, "search_width": 8}
            ],
            'GPU_IVF_FLAT': [
                {"nprobe": 10},
                {"nprobe": 50},
                {"nprobe": 100},
                {"nprobe": 500}
            ],
            'GPU_IVF_PQ': [
                {"nprobe": 10},
                {"nprobe": 50},
                {"nprobe": 100},
                {"nprobe": 500}
            ],
            'GPU_BRUTE_FORCE': [
                {}  # No parameters needed
            ]
        }
        
        results = {
            'num_searches': num_searches,
            'topk': topk,
            'parameter_results': []
        }
        
        # Test each search parameter
        for search_param in search_params.get(index_type, [{"nprobe": 10}]):
            # Warm-up
            self.client.search(
                collection_name=collection_name,
                data=[target_embedding],
                anns_field="embedding",
                param={
                    "metric_type": "L2",
                    "params": search_param
                },
                limit=topk,
                output_fields=["text"]
            )
            
            start_time = time.time()
            found_count = 0
            positions = []  # Track positions when found
            
            for _ in range(num_searches):
                search_results = self.client.search(
                    collection_name=collection_name,
                    data=[target_embedding],
                    anns_field="embedding",
                    param={
                        "metric_type": "L2",
                        "params": search_param
                    },
                    limit=topk,
                    output_fields=["text"]
                )
                
                # Find position of target_id in results
                found = False
                for pos, hit in enumerate(search_results[0]):
                    if hit["id"] == target_id:  # Adjust based on actual result format
                        found = True
                        found_count += 1
                        positions.append(pos + 1)  # Add 1 for 1-based position
                        break
                if not found:
                    positions.append(None)  # Mark as not found
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate average position when found
            found_positions = [p for p in positions if p is not None]
            avg_position = sum(found_positions) / len(found_positions) if found_positions else None
            
            param_result = {
                'parameters': search_param,
                'total_time': total_time,
                'searches_per_second': num_searches / total_time,
                'avg_time_per_search': total_time / num_searches,
                'target_found_rate': found_count / num_searches,
                'avg_position_when_found': avg_position,
                'position_stats': {
                    'min': min(found_positions) if found_positions else None,
                    'max': max(found_positions) if found_positions else None,
                    'positions': positions
                }
            }
            results['parameter_results'].append(param_result)
        
        return results

    def plant_search_data(self, collection_name, target_text, target_embedding):
        # Get the current count of entities
        stats = self.client.get_collection_stats(collection_name)
        row_count = stats["row_count"]
        
        # Insert the target data
        self.client.insert(
            collection_name=collection_name,
            data=[{
                "id": row_count,
                "text": target_text,
                "embedding": target_embedding.tolist()
            }]
        )
        
        # Flush to ensure data is persisted
        self.client.flush(collection_name=collection_name)
        
        # Return the ID of the inserted entity
        return row_count

    def estimate_collection_size(self, num_vectors, dim=1536, text_avg_length=100):
        """Estimate the size of the collection in memory and on disk"""
        # Estimate vector data size (4 bytes per float)
        vector_size_bytes = num_vectors * dim * 4
        
        # Estimate text data size (2 bytes per character on average)
        text_size_bytes = num_vectors * text_avg_length * 2
        
        # Estimate primary key size (8 bytes per int64)
        pk_size_bytes = num_vectors * 8
        
        # Total raw data size
        raw_data_size_bytes = vector_size_bytes + text_size_bytes + pk_size_bytes
        
        # Estimate index sizes for different types (CPU)
        index_sizes = {
            'IVF_FLAT': vector_size_bytes * 1.1,  # IVF_FLAT adds ~10% overhead
            'IVF_SQ8': vector_size_bytes * 0.3,   # IVF_SQ8 compresses to ~30% of original
            'HNSW': vector_size_bytes * 1.5       # HNSW adds ~50% overhead
        }
        
        # Estimate index sizes for different types (GPU)
        # Based on the documentation: https://milvus.io/docs/gpu_index.md
        gpu_index_sizes = {
            'GPU_CAGRA': vector_size_bytes * 1.8,     # ~1.8x original data size
            'GPU_IVF_FLAT': vector_size_bytes * 1.0,  # Same as original data
            'GPU_IVF_PQ': vector_size_bytes * 0.3,    # Compressed like CPU IVF_PQ
            'GPU_BRUTE_FORCE': vector_size_bytes * 1.0  # Same as original data
        }
        
        # Combine both dictionaries
        index_sizes.update(gpu_index_sizes)
        
        return {
            'raw_data_size_mb': raw_data_size_bytes / (1024 * 1024),
            'index_sizes_mb': {k: v / (1024 * 1024) for k, v in index_sizes.items()}
        }

    def run_complete_benchmark(self, collection_sizes=[100, 1000, 5000],
                             dim=1536, target_text="This is a specific text that we want to find later"):
        # Only get embedding for the target text we want to search for
        print("Getting embedding for target text...")
        target_embedding = self.get_embedding([target_text])[0]
           
        results = {
            'collection_sizes': {},
            'parameters': {
                'dimensions': dim,
                'target_text': target_text,
                'model_name': "OpenAI (search only)"
            },
            'estimated_sizes': {}
        }
        
        # Add size estimates to results
        for size in collection_sizes:
            results['estimated_sizes'][size] = self.estimate_collection_size(size, dim)
       
        for size in collection_sizes:
            print(f"\nTesting collection size: {size}")
            size_results = {'index_types': {}}
           
            for index_type in self.index_configs.keys():
                print(f"\nTesting index: {index_type}")
                collection_name, index_time_empty = self.create_collection(dim=dim, index_type=index_type)
               
                insert_results = self.benchmark_insertion(collection_name, batch_sizes=[5000], total_vectors=size, dim=dim, index_type=index_type)
               
                # Plant our target text with its real embedding
                target_id = self.plant_search_data(collection_name, target_text, target_embedding)
                search_results = self.benchmark_search(collection_name, target_embedding, target_id)
               
                results_for_size = {
                    'insertion': insert_results[0],
                    'search': search_results,
                    'index_time_empty': index_time_empty,
                    'total_vectors': size,
                    'estimated_size_mb': results['estimated_sizes'][size]['raw_data_size_mb'],
                    'estimated_index_size_mb': results['estimated_sizes'][size]['index_sizes_mb'][index_type]
                }
                size_results['index_types'][index_type] = results_for_size
               
                self.client.drop_collection(collection_name)
           
            results['collection_sizes'][size] = size_results
       
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=4)
       
        self.plot_results(results)
        return results

    def plot_results(self, results):
        collection_sizes = list(results['collection_sizes'].keys())
        index_types = list(results['collection_sizes'][collection_sizes[0]]['index_types'].keys())
        
        plt.figure(figsize=(20, 24))  # Made taller for 6 plots
        
        # Plot insertion times (not speeds)
        plt.subplot(3, 2, 1)
        for idx_type in index_types:
            times = [results['collection_sizes'][size]['index_types'][idx_type]['insertion']['insert_time']
                    for size in collection_sizes]
            plt.plot(collection_sizes, times, marker='o', label=idx_type)
        plt.title('Insertion Time')
        plt.xlabel('Collection Size (vectors)')
        plt.ylabel('Seconds')
        plt.legend()
        plt.xscale('log')
        
        # Plot index creation times
        plt.subplot(3, 2, 2)
        for idx_type in index_types:
            times = [results['collection_sizes'][size]['index_types'][idx_type]['insertion']['index_time_after_insert']
                    for size in collection_sizes]
            plt.plot(collection_sizes, times, marker='o', label=idx_type)
        plt.title('Index Creation Time')
        plt.xlabel('Collection Size (vectors)')
        plt.ylabel('Seconds')
        plt.legend()
        plt.xscale('log')
        
        # Plot search performance
        plt.subplot(3, 2, 3)
        for idx_type in index_types:
            for size in collection_sizes:
                search_results = results['collection_sizes'][size]['index_types'][idx_type]['search']['parameter_results']
                x_labels = [str(p['parameters']) for p in search_results]
                speeds = [p['searches_per_second'] for p in search_results]
                plt.plot(x_labels, speeds, marker='o', label=f"{idx_type} ({size} vectors)")
        plt.title('Search Performance by Parameter')
        plt.xlabel('Search Parameters')
        plt.ylabel('Searches per Second')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot search accuracy
        plt.subplot(3, 2, 4)
        for idx_type in index_types:
            for size in collection_sizes:
                search_results = results['collection_sizes'][size]['index_types'][idx_type]['search']['parameter_results']
                x_labels = [str(p['parameters']) for p in search_results]
                accuracy = [p['target_found_rate'] * 100 for p in search_results]
                plt.plot(x_labels, accuracy, marker='o', label=f"{idx_type} ({size} vectors)")
        plt.title('Search Accuracy by Parameter')
        plt.xlabel('Search Parameters')
        plt.ylabel('Target Found Rate (%)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add plot for average position
        plt.subplot(3, 2, 5)
        for idx_type in index_types:
            for size in collection_sizes:
                search_results = results['collection_sizes'][size]['index_types'][idx_type]['search']['parameter_results']
                x_labels = [str(p['parameters']) for p in search_results]
                positions = [p['avg_position_when_found'] for p in search_results]
                plt.plot(x_labels, positions, marker='o', label=f"{idx_type} ({size} vectors)")
        plt.title('Average Position of Target in Results')
        plt.xlabel('Search Parameters')
        plt.ylabel('Average Position (lower is better)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add estimated collection size plot
        plt.subplot(3, 2, 6)
        bar_width = 0.2
        x = np.arange(len(collection_sizes))
        
        for i, idx_type in enumerate(index_types):
            sizes = [results['estimated_sizes'][size]['index_sizes_mb'][idx_type] for size in collection_sizes]
            plt.bar(x + i*bar_width, sizes, width=bar_width, label=f"{idx_type} Index")
        
        # Add raw data size
        raw_sizes = [results['estimated_sizes'][size]['raw_data_size_mb'] for size in collection_sizes]
        plt.bar(x + len(index_types)*bar_width, raw_sizes, width=bar_width, label="Raw Data")
        
        plt.title('Estimated Collection Size')
        plt.xlabel('Collection Size (vectors)')
        plt.ylabel('Size (MB)')
        plt.xticks(x + bar_width*1.5, collection_sizes)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png', bbox_inches='tight')
        plt.close()

    def benchmark_concurrent_search(self, collection_name, target_embedding, num_concurrent=10, 
                                   num_searches_per_thread=10, topk=100, index_type="IVF_FLAT"):
        """
        Benchmark concurrent search performance
        
        Args:
            collection_name: Name of the collection to search
            target_embedding: Embedding vector to search for
            num_concurrent: Number of concurrent search threads
            num_searches_per_thread: Number of searches each thread will perform
            topk: Number of results to return per search
            index_type: Type of index being used
        
        Returns:
            Dictionary with benchmark results
        """
        self.client.load_collection(collection_name)
        
        # Get appropriate search parameters based on index type
        if index_type in self.index_configs:
            if index_type == "IVF_FLAT" or index_type == "IVF_SQ8":
                search_params = {"nprobe": 50}
            elif index_type == "HNSW":
                search_params = {"ef": 50}
        else:
            search_params = {"nprobe": 50}  # Default
        
        # Shared results container with thread-safe access
        results_lock = threading.Lock()
        search_times = []
        success_count = 0
        
        def search_worker():
            nonlocal success_count
            local_times = []
            local_success = 0
            
            for _ in range(num_searches_per_thread):
                start_time = time.time()
                try:
                    search_results = self.client.search(
                        collection_name=collection_name,
                        data=[target_embedding],
                        anns_field="embedding",
                        param={
                            "metric_type": "L2",
                            "params": search_params
                        },
                        limit=topk,
                        output_fields=["text"]
                    )
                    end_time = time.time()
                    local_times.append(end_time - start_time)
                    local_success += 1
                except Exception as e:
                    print(f"Search error: {e}")
            
            # Update shared results
            with results_lock:
                search_times.extend(local_times)
                success_count += local_success
        
        # Create and start threads
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(search_worker) for _ in range(num_concurrent)]
            concurrent.futures.wait(futures)
        
        total_time = time.time() - start_time
        total_searches = num_concurrent * num_searches_per_thread
        
        # Calculate statistics
        if search_times:
            avg_search_time = sum(search_times) / len(search_times)
            min_search_time = min(search_times)
            max_search_time = max(search_times)
            p95_search_time = sorted(search_times)[int(len(search_times) * 0.95)]
        else:
            avg_search_time = min_search_time = max_search_time = p95_search_time = 0
        
        return {
            'num_concurrent': num_concurrent,
            'num_searches_per_thread': num_searches_per_thread,
            'total_searches': total_searches,
            'successful_searches': success_count,
            'success_rate': success_count / total_searches if total_searches > 0 else 0,
            'total_time': total_time,
            'searches_per_second': total_searches / total_time if total_time > 0 else 0,
            'avg_search_time': avg_search_time,
            'min_search_time': min_search_time,
            'max_search_time': max_search_time,
            'p95_search_time': p95_search_time,
            'search_params': search_params
        }

    def run_concurrent_benchmark(self, collection_sizes=[1000, 5000], 
                                concurrent_levels=[1, 10, 50, 100, 200, 500, 1000],
                                searches_per_thread=10,
                                dim=1536, 
                                target_text="This is a specific text that we want to find later"):
        """
        Run a comprehensive concurrent search benchmark
        
        This benchmark focuses exclusively on search performance under concurrent load.
        For each collection size and index type, it tests how the system performs with
        different numbers of concurrent search requests.
        
        Args:
            collection_sizes: List of collection sizes to test
            concurrent_levels: List of concurrent thread counts to test
            searches_per_thread: Number of searches each thread will perform
            dim: Vector dimension
            target_text: Text to search for
        
        Returns:
            Dictionary with benchmark results
        """
        print("Getting embedding for target text...")
        target_embedding = self.get_embedding([target_text])[0]
        
        results = {
            'collection_sizes': {},
            'parameters': {
                'dimensions': dim,
                'target_text': target_text,
                'model_name': "OpenAI (concurrent search)",
                'searches_per_thread': searches_per_thread,
                'concurrent_levels_tested': concurrent_levels
            }
        }
        
        # For each collection size
        for size in collection_sizes:
            print(f"\n=== Testing collection size: {size} vectors ===")
            size_results = {'index_types': {}}
            
            # For each index type
            for index_type in self.index_configs.keys():
                print(f"\n--- Testing index: {index_type} ---")
                
                # Create collection and insert data only once per index type and size
                collection_name, _ = self.create_collection(dim=dim, index_type=index_type)
                
                # Insert random vectors
                print(f"Inserting {size} vectors...")
                insert_results = self.benchmark_insertion(collection_name, batch_sizes=[5000], total_vectors=size, dim=dim, index_type=index_type)
                
                # Plant our target text with its real embedding
                target_id = self.plant_search_data(collection_name, target_text, target_embedding)
                
                # Ensure collection is loaded before concurrent testing
                print("Loading collection into memory...")
                self.client.load_collection(collection_name)
                
                # Test different concurrency levels
                concurrency_results = []
                print(f"Running concurrent search tests with {len(concurrent_levels)} concurrency levels...")
                
                for concurrent in concurrent_levels:
                    print(f"  Testing with {concurrent} concurrent threads...")
                    result = self.benchmark_concurrent_search(
                        collection_name, 
                        target_embedding, 
                        num_concurrent=concurrent,
                        num_searches_per_thread=searches_per_thread,
                        index_type=index_type
                    )
                    concurrency_results.append(result)
                    
                    # Print some immediate feedback
                    print(f"    → {result['searches_per_second']:.2f} searches/sec, " 
                          f"avg: {result['avg_search_time']*1000:.2f}ms, "
                          f"p95: {result['p95_search_time']*1000:.2f}ms")
                
                size_results['index_types'][index_type] = {
                    'collection_size': size,
                    'index_type': index_type,
                    'insertion_time': insert_results[0],
                    'index_creation_time': insert_results[0]['index_time_after_insert'],
                    'concurrent_search': concurrency_results
                }
                
                # Release collection from memory before dropping
                self.client.release_collection(collection_name)
                self.client.drop_collection(collection_name)
                
                # Save intermediate results after each index type
                with open(f'concurrent_benchmark_results_{size}_{index_type}.json', 'w') as f:
                    json.dump(size_results['index_types'][index_type], f, indent=4)
                
            results['collection_sizes'][size] = size_results
        
        # Save complete results
        with open('concurrent_benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generate plots
        self.plot_concurrent_results(results, concurrent_levels)
        return results

    def plot_concurrent_results(self, results, concurrent_levels):
        """
        Plot the results of concurrent benchmarks
        
        Args:
            results: Results dictionary from run_concurrent_benchmark
            concurrent_levels: List of concurrent thread counts tested
        """
        collection_sizes = list(results['collection_sizes'].keys())
        index_types = list(results['collection_sizes'][collection_sizes[0]]['index_types'].keys())
        
        # Create a figure with two rows of plots
        plt.figure(figsize=(20, 16))
        
        # Plot 1: Searches per second vs Concurrency level (by index type)
        plt.subplot(2, 2, 1)
        for idx_type in index_types:
            for size in collection_sizes:
                concurrency_results = results['collection_sizes'][size]['index_types'][idx_type]['concurrent_search']
                x_values = [r['num_concurrent'] for r in concurrency_results]
                y_values = [r['searches_per_second'] for r in concurrency_results]
                plt.plot(x_values, y_values, marker='o', label=f"{idx_type} ({size} vectors)")
        
        plt.title('Search Throughput vs Concurrency')
        plt.xlabel('Number of Concurrent Threads')
        plt.ylabel('Searches per Second (higher is better)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')  # Log scale for x-axis to better show the range
        
        # Plot 2: Average search time vs Concurrency level
        plt.subplot(2, 2, 2)
        for idx_type in index_types:
            for size in collection_sizes:
                concurrency_results = results['collection_sizes'][size]['index_types'][idx_type]['concurrent_search']
                x_values = [r['num_concurrent'] for r in concurrency_results]
                y_values = [r['avg_search_time'] * 1000 for r in concurrency_results]  # Convert to ms
                plt.plot(x_values, y_values, marker='o', label=f"{idx_type} ({size} vectors)")
        
        plt.title('Average Search Time vs Concurrency')
        plt.xlabel('Number of Concurrent Threads')
        plt.ylabel('Average Search Time (ms) (lower is better)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        # Plot 3: P95 search time vs Concurrency level
        plt.subplot(2, 2, 3)
        for idx_type in index_types:
            for size in collection_sizes:
                concurrency_results = results['collection_sizes'][size]['index_types'][idx_type]['concurrent_search']
                x_values = [r['num_concurrent'] for r in concurrency_results]
                y_values = [r['p95_search_time'] * 1000 for r in concurrency_results]  # Convert to ms
                plt.plot(x_values, y_values, marker='o', label=f"{idx_type} ({size} vectors)")
        
        plt.title('P95 Search Time vs Concurrency')
        plt.xlabel('Number of Concurrent Threads')
        plt.ylabel('P95 Search Time (ms) (lower is better)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        # Plot 4: Throughput efficiency (searches per second / concurrency)
        plt.subplot(2, 2, 4)
        for idx_type in index_types:
            for size in collection_sizes:
                concurrency_results = results['collection_sizes'][size]['index_types'][idx_type]['concurrent_search']
                x_values = [r['num_concurrent'] for r in concurrency_results]
                # Calculate efficiency: searches per second divided by number of concurrent threads
                # This shows how efficiently the system scales with more threads
                y_values = [r['searches_per_second'] / r['num_concurrent'] for r in concurrency_results]
                plt.plot(x_values, y_values, marker='o', label=f"{idx_type} ({size} vectors)")
        
        plt.title('Throughput Efficiency vs Concurrency')
        plt.xlabel('Number of Concurrent Threads')
        plt.ylabel('Searches/Second per Thread (higher is better)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig('concurrent_benchmark_results.png', bbox_inches='tight')
        
        # Create a second figure for additional plots
        plt.figure(figsize=(20, 10))
        
        # Plot 5: Max search time vs Concurrency level
        plt.subplot(1, 2, 1)
        for idx_type in index_types:
            for size in collection_sizes:
                concurrency_results = results['collection_sizes'][size]['index_types'][idx_type]['concurrent_search']
                x_values = [r['num_concurrent'] for r in concurrency_results]
                y_values = [r['max_search_time'] * 1000 for r in concurrency_results]  # Convert to ms
                plt.plot(x_values, y_values, marker='o', label=f"{idx_type} ({size} vectors)")
        
        plt.title('Maximum Search Time vs Concurrency')
        plt.xlabel('Number of Concurrent Threads')
        plt.ylabel('Maximum Search Time (ms)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        # Plot 6: Total time to complete all searches
        plt.subplot(1, 2, 2)
        for idx_type in index_types:
            for size in collection_sizes:
                concurrency_results = results['collection_sizes'][size]['index_types'][idx_type]['concurrent_search']
                x_values = [r['num_concurrent'] for r in concurrency_results]
                y_values = [r['total_time'] for r in concurrency_results]
                plt.plot(x_values, y_values, marker='o', label=f"{idx_type} ({size} vectors)")
        
        plt.title('Total Time to Complete All Searches')
        plt.xlabel('Number of Concurrent Threads')
        plt.ylabel('Total Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig('concurrent_benchmark_additional.png', bbox_inches='tight')
        plt.close('all')

    def benchmark_memory_usage(self, collection_name, size, index_type):
        """
        Benchmark memory usage of a collection
        
        Note: This is an estimation as Milvus doesn't provide direct memory usage metrics
        through the Python SDK. For more accurate measurements, you would need to monitor
        the Milvus process using system tools.
        
        Args:
            collection_name: Name of the collection
            size: Number of vectors in collection
            index_type: Type of index used
            
        Returns:
            Dictionary with memory usage estimates
        """
        # Get collection stats
        stats = self.client.get_collection_stats(collection_name)
        row_count = stats["row_count"]
        
        # Estimate memory usage based on collection size and index type
        dim = 1536  # OpenAI embedding dimension
        
        # Vector data size (4 bytes per float)
        vector_data_size = row_count * dim * 4
        
        # Index memory overhead estimates
        index_memory_multipliers = {
            'IVF_FLAT': 1.1,  # ~10% overhead
            'IVF_SQ8': 0.3,   # ~70% compression
            'HNSW': 1.5       # ~50% overhead
        }
        
        index_memory = vector_data_size * index_memory_multipliers.get(index_type, 1.0)
        
        # Total memory estimate
        total_memory = vector_data_size + index_memory
        
        return {
            'row_count': row_count,
            'vector_data_size_mb': vector_data_size / (1024 * 1024),
            'index_memory_mb': index_memory / (1024 * 1024),
            'total_memory_mb': total_memory / (1024 * 1024),
            'memory_per_vector_kb': (total_memory / row_count) / 1024 if row_count > 0 else 0
        }

    def benchmark_recall(self, collection_name, num_queries=100, topk=100, ground_truth_method="exhaustive"):
        """
        Benchmark recall rate of the index
        
        This measures how well the approximate search performs compared to an exhaustive search.
        
        Args:
            collection_name: Name of the collection
            num_queries: Number of random queries to perform
            topk: Number of results to return
            ground_truth_method: Method to obtain ground truth ("exhaustive" or "brute_force")
            
        Returns:
            Dictionary with recall metrics
        """
        self.client.load_collection(collection_name)
        
        # Get index type
        try:
            index_desc = self.client.describe_index(
                collection_name=collection_name,
                index_name="embedding"  # Index name is the same as field name by default
            )
            # The structure is different in the new API - directly access index_type
            index_type = index_desc['index_type'] if index_desc else "NONE"
        except Exception as e:
            print(f"Error getting index type: {e}")
            index_type = "NONE"
        
        # Generate random query vectors
        query_vectors = np.random.rand(num_queries, 1536).astype(np.float32)
        
        # Get ground truth results using exhaustive search
        ground_truth_results = []
        if ground_truth_method == "exhaustive":
            # Use IVF_FLAT with nprobe set to a high value for near-exhaustive search
            for qvec in tqdm(query_vectors, desc="Getting ground truth"):
                results = self.client.search(
                    collection_name=collection_name,
                    data=[qvec],
                    anns_field="embedding",
                    param={"metric_type": "L2", "params": {"nprobe": 1024}},  # High nprobe for exhaustive search
                    limit=topk * 2,  # Get more results to ensure we have enough ground truth
                    output_fields=["id"]
                )
                ground_truth_results.append([hit["id"] for hit in results[0]])
        
        # Test different search parameters
        recall_results = []
        
        # Define search parameters for different index types
        search_params = {
            'IVF_FLAT': [
                {"nprobe": 10},
                {"nprobe": 50},
                {"nprobe": 100},
                {"nprobe": 500}
            ],
            'IVF_SQ8': [
                {"nprobe": 10},
                {"nprobe": 50},
                {"nprobe": 100},
                {"nprobe": 500}
            ],
            'HNSW': [
                {"ef": 10},
                {"ef": 50},
                {"ef": 100},
                {"ef": 500}
            ],
            'NONE': [{}]  # No parameters for brute force
        }
        
        for param in search_params.get(index_type, [{}]):
            recall_at_k = []
            search_times = []
            
            for i, qvec in enumerate(tqdm(query_vectors, desc=f"Testing {param}")):
                start_time = time.time()
                results = self.client.search(
                    collection_name=collection_name,
                    data=[qvec],
                    anns_field="embedding",
                    param={
                        "metric_type": "L2",
                        "params": param
                    },
                    limit=topk,
                    output_fields=["id"]
                )
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                # Calculate recall@k
                result_ids = [hit["id"] for hit in results[0]]
                ground_truth_ids = ground_truth_results[i][:topk]
                
                # Count how many of the ground truth results are in our results
                matches = len(set(result_ids) & set(ground_truth_ids))
                recall = matches / len(ground_truth_ids) if ground_truth_ids else 0
                recall_at_k.append(recall)
            
            recall_results.append({
                'parameters': param,
                'avg_recall': sum(recall_at_k) / len(recall_at_k),
                'min_recall': min(recall_at_k),
                'max_recall': max(recall_at_k),
                'avg_search_time': sum(search_times) / len(search_times),
                'searches_per_second': len(search_times) / sum(search_times)
            })
        
        return {
            'num_queries': num_queries,
            'topk': topk,
            'index_type': index_type,
            'recall_results': recall_results
        }

    def benchmark_latency_distribution(self, collection_name, target_embedding, num_searches=1000):
        """
        Benchmark the distribution of search latencies
        
        This provides percentile metrics for search latency.
        
        Args:
            collection_name: Name of the collection
            target_embedding: Embedding vector to search for
            num_searches: Number of searches to perform
            
        Returns:
            Dictionary with latency distribution metrics
        """
        self.client.load_collection(collection_name)
        
        # Get index type
        try:
            index_desc = self.client.describe_index(
                collection_name=collection_name,
                index_name="embedding"  # Index name is the same as field name by default
            )
            # The structure is different in the new API - directly access index_type
            index_type = index_desc['index_type'] if index_desc else "NONE"
        except Exception as e:
            print(f"Error getting index type: {e}")
            index_type = "NONE"
        
        # Define search parameters based on index type
        if index_type in ["IVF_FLAT", "IVF_SQ8"]:
            search_params = {"nprobe": 50}
        elif index_type == "HNSW":
            search_params = {"ef": 50}
        else:
            search_params = {}
        
        # Perform searches and record latencies
        latencies = []
        
        for _ in tqdm(range(num_searches), desc="Measuring search latencies"):
            start_time = time.time()
            results = self.client.search(
                collection_name=collection_name,
                data=[target_embedding],
                anns_field="embedding",
                param={
                    "metric_type": "L2",
                    "params": search_params
                },
                limit=10,  # topk
                output_fields=["text"]
            )
            latency = time.time() - start_time
            latencies.append(latency * 1000)  # Convert to ms
        
        # Calculate percentiles
        latencies.sort()
        percentiles = {
            'p50': latencies[int(num_searches * 0.5)],
            'p90': latencies[int(num_searches * 0.9)],
            'p95': latencies[int(num_searches * 0.95)],
            'p99': latencies[int(num_searches * 0.99)],
            'min': latencies[0],
            'max': latencies[-1],
            'avg': sum(latencies) / num_searches
        }
        
        return {
            'num_searches': num_searches,
            'index_type': index_type,
            'search_params': search_params,
            'latency_percentiles_ms': percentiles,
            'latency_distribution': latencies
        }

    def run_unified_benchmark(self, collection_sizes=[100, 1000, 5000, 10000],
                             concurrent_levels=[1, 10, 50, 100, 200, 500],
                             dim=1536, 
                             target_text="This is a specific text that we want to find later",
                             resume=True,
                             use_gpu=False):
        """
        Run a unified benchmark that combines all test types
        
        This benchmark creates each collection only once and runs all tests on it.
        It can resume from previous runs by loading existing result files.
        
        Args:
            collection_sizes: List of collection sizes to test
            concurrent_levels: List of concurrent thread counts to test
            dim: Vector dimension
            target_text: Text to search for
            resume: Whether to resume from previous runs
            use_gpu: Whether to use GPU indexes
            
        Returns:
            Dictionary with all benchmark results
        """
        print("Getting embedding for target text...")
        target_embedding = self.get_embedding([target_text])[0]
        
        # Choose which index configurations to use
        index_configs = self.gpu_index_configs if use_gpu else self.index_configs
        index_type_prefix = "GPU_" if use_gpu else ""
        
        # Special handling for GPU_IVF_PQ's 'm' parameter based on dimension
        if use_gpu and 'GPU_IVF_PQ' in index_configs:
            # For GPU_IVF_PQ, m must be a divisor of the dimension
            # Find the largest divisor of dim that is <= 64
            for m in range(min(64, dim), 0, -1):
                if dim % m == 0:
                    index_configs['GPU_IVF_PQ']['params']['m'] = m
                    break
        
        # Initialize results structure
        results = {
            'collection_sizes': {},
            'parameters': {
                'dimensions': dim,
                'target_text': target_text,
                'model_name': f"OpenAI ({'GPU' if use_gpu else 'CPU'} unified benchmark)",
                'concurrent_levels_tested': concurrent_levels,
                'use_gpu': use_gpu
            },
            'estimated_sizes': {}
        }
        
        # Check if we have a previous complete run
        results_file = f"unified_benchmark_results_{'gpu' if use_gpu else 'cpu'}.json"
        if resume and os.path.exists(results_file):
            try:
                print(f"Found previous benchmark results. Loading from {results_file}...")
                with open(results_file, 'r') as f:
                    previous_results = json.load(f)
                    
                # Use previous results as a starting point
                results = previous_results
                print("Successfully loaded previous results.")
            except Exception as e:
                print(f"Error loading previous results: {e}")
                print("Starting fresh benchmark.")
        
        # Add size estimates to results if not already present
        if 'estimated_sizes' not in results or not results['estimated_sizes']:
            results['estimated_sizes'] = {}
            for size in collection_sizes:
                # Always use string keys for the dictionary
                size_str = str(size)
                results['estimated_sizes'][size_str] = self.estimate_collection_size(size, dim)
        
        # For each collection size
        for size in collection_sizes:
            # Always use string keys for the dictionary
            size_str = str(size)
            print(f"\n=== Testing collection size: {size} vectors ===")
            
            # Initialize size results if not present
            if size_str not in results['collection_sizes']:
                results['collection_sizes'][size_str] = {'index_types': {}}
            
            size_results = results['collection_sizes'][size_str]
            
            # For each index type
            for index_type in index_configs.keys():
                print(f"\n--- Testing index: {index_type} ---")
                
                # Check if we already have results for this combination
                result_file = f"unified_benchmark_{size}_{index_type}_{'gpu' if use_gpu else 'cpu'}.json"
                if resume and os.path.exists(result_file):
                    try:
                        print(f"Found existing results for size {size}, index {index_type}. Loading...")
                        with open(result_file, 'r') as f:
                            existing_results = json.load(f)
                        
                        # Add to our results and skip this combination
                        size_results['index_types'][index_type] = existing_results
                        print(f"Skipping tests for size {size}, index {index_type} - using existing results.")
                        continue
                    except Exception as e:
                        print(f"Error loading existing results: {e}")
                        print(f"Will run tests for size {size}, index {index_type}")
                
                # Create collection and insert data
                collection_name = f"benchmark_{size}_{index_type}"
                print(f"Creating collection {collection_name}...")
                collection_name, index_time_empty = self.create_collection(
                    dim=dim, 
                    collection_name=collection_name,
                    index_type=index_type,
                    index_config=index_configs[index_type]
                )
                
                # Insert random vectors
                print(f"Inserting {size} vectors...")
                insert_results = self.benchmark_insertion(
                    collection_name,
                    batch_sizes=[5000],
                    total_vectors=size,
                    dim=dim,
                    index_type=index_type
                )
                
                # Plant our target text with its real embedding
                print("Planting target text...")
                target_id = self.plant_search_data(collection_name, target_text, target_embedding)
                
                # 1. Standard search benchmark
                print("Running standard search benchmark...")
                search_results = self.benchmark_search(collection_name, target_embedding, target_id)
                
                # 2. Memory usage benchmark
                print("Estimating memory usage...")
                memory_results = self.benchmark_memory_usage(collection_name, size, index_type)
                
                # 3. Latency distribution benchmark
                print("Measuring latency distribution...")
                latency_results = self.benchmark_latency_distribution(
                    collection_name, 
                    target_embedding, 
                    num_searches=100  # Reduced for efficiency
                )
                
                # 4. Concurrent search benchmark
                print("Running concurrent search tests...")
                concurrency_results = []
                for concurrent in concurrent_levels:
                    print(f"  Testing with {concurrent} concurrent threads...")
                    result = self.benchmark_concurrent_search(
                        collection_name, 
                        target_embedding, 
                        num_concurrent=concurrent,
                        num_searches_per_thread=5,  # Reduced for efficiency
                        index_type=index_type
                    )
                    concurrency_results.append(result)
                    
                    # Print some immediate feedback
                    print(f"    → {result['searches_per_second']:.2f} searches/sec, " 
                          f"avg: {result['avg_search_time']*1000:.2f}ms, "
                          f"p95: {result['p95_search_time']*1000:.2f}ms")
                
                # 5. Recall benchmark (only for larger collections)
                recall_results = None
                if size >= 1000:  # Lower threshold to include more collections
                    print("Measuring recall rates...")
                    recall_results = self.benchmark_recall(
                        collection_name, 
                        num_queries=10,  # Reduced for efficiency
                        topk=100
                    )
                
                # Store all results - ensure we're using string keys consistently
                current_results = {
                    'collection_size': size,
                    'index_type': index_type,
                    'insertion': insert_results[0],
                    'search': search_results,
                    'memory': memory_results,
                    'latency_distribution': latency_results,
                    'concurrent_search': concurrency_results,
                    'recall': recall_results,
                    'index_time_empty': index_time_empty,
                    'estimated_size_mb': results['estimated_sizes'][size_str]['raw_data_size_mb'],
                    'estimated_index_size_mb': results['estimated_sizes'][size_str]['index_sizes_mb'][index_type]
                }
                
                # Save individual results file (checkpoint)
                with open(result_file, 'w') as f:
                    json.dump(current_results, f, indent=4)
                
                # Add to our results
                size_results['index_types'][index_type] = current_results
                
                # Save complete results after each index type (additional checkpoint)
                with open('unified_benchmark_results.json', 'w') as f:
                    json.dump(results, f, indent=4)
                
                # Release and drop collection
                print(f"Releasing and dropping collection {collection_name}...")
                self.client.release_collection(collection_name)
                self.client.drop_collection(collection_name)
        
        # Generate plots
        print("Generating plots...")
        self.plot_unified_results(results, concurrent_levels)
        return results

    def plot_unified_results(self, results, concurrent_levels):
        """
        Plot unified benchmark results
        
        Args:
            results: Results dictionary from run_unified_benchmark
            concurrent_levels: List of concurrent thread counts tested
        """
        collection_sizes = list(results['collection_sizes'].keys())
        index_types = list(results['collection_sizes'][collection_sizes[0]]['index_types'].keys())
        
        # Create a multi-page figure for basic metrics
        plt.figure(figsize=(20, 24))
        
        # Page 1: Basic metrics
        # Plot 1: Insertion time
        plt.subplot(3, 2, 1)
        for idx_type in index_types:
            times = [results['collection_sizes'][size]['index_types'][idx_type]['insertion']['insert_time']
                    for size in collection_sizes]
            plt.plot(collection_sizes, times, marker='o', label=idx_type)
        plt.title('Insertion Time')
        plt.xlabel('Collection Size (vectors)')
        plt.ylabel('Seconds')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Index creation time
        plt.subplot(3, 2, 2)
        for idx_type in index_types:
            times = [results['collection_sizes'][size]['index_types'][idx_type]['insertion']['index_time_after_insert']
                    for size in collection_sizes]
            plt.plot(collection_sizes, times, marker='o', label=idx_type)
        plt.title('Index Creation Time')
        plt.xlabel('Collection Size (vectors)')
        plt.ylabel('Seconds')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Memory usage
        plt.subplot(3, 2, 3)
        for idx_type in index_types:
            memory = [results['collection_sizes'][size]['index_types'][idx_type]['memory']['total_memory_mb']
                     for size in collection_sizes]
            plt.plot(collection_sizes, memory, marker='o', label=idx_type)
        plt.title('Memory Usage')
        plt.xlabel('Collection Size (vectors)')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Search performance by parameter (for largest collection size)
        plt.subplot(3, 2, 4)
        largest_size = max(collection_sizes)
        for idx_type in index_types:
            search_results = results['collection_sizes'][largest_size]['index_types'][idx_type]['search']['parameter_results']
            x_labels = [str(p['parameters']) for p in search_results]
            speeds = [p['searches_per_second'] for p in search_results]
            plt.plot(x_labels, speeds, marker='o', label=idx_type)
        plt.title(f'Search Performance by Parameter (Size: {largest_size})')
        plt.xlabel('Search Parameters')
        plt.ylabel('Searches per Second')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Plot 5: Search accuracy by parameter (for largest collection size)
        plt.subplot(3, 2, 5)
        for idx_type in index_types:
            search_results = results['collection_sizes'][largest_size]['index_types'][idx_type]['search']['parameter_results']
            x_labels = [str(p['parameters']) for p in search_results]
            accuracy = [p['target_found_rate'] * 100 for p in search_results]
            plt.plot(x_labels, accuracy, marker='o', label=idx_type)
        plt.title(f'Search Accuracy by Parameter (Size: {largest_size})')
        plt.xlabel('Search Parameters')
        plt.ylabel('Target Found Rate (%)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Plot 6: Estimated collection size
        plt.subplot(3, 2, 6)
        bar_width = 0.2
        x = np.arange(len(collection_sizes))
        
        for i, idx_type in enumerate(index_types):
            sizes = [results['estimated_sizes'][size]['index_sizes_mb'][idx_type] for size in collection_sizes]
            plt.bar(x + i*bar_width, sizes, width=bar_width, label=f"{idx_type} Index")
        
        # Add raw data size
        raw_sizes = [results['estimated_sizes'][size]['raw_data_size_mb'] for size in collection_sizes]
        plt.bar(x + len(index_types)*bar_width, raw_sizes, width=bar_width, label="Raw Data")
        
        plt.title('Estimated Collection Size')
        plt.xlabel('Collection Size (vectors)')
        plt.ylabel('Size (MB)')
        plt.xticks(x + bar_width*1.5, collection_sizes)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('unified_benchmark_basic.png', bbox_inches='tight')
        
        # Create a second figure for concurrency metrics
        plt.figure(figsize=(20, 15))
        
        # Plot 1: Searches per second vs Concurrency level
        plt.subplot(2, 2, 1)
        for idx_type in index_types:
            for size in collection_sizes:
                concurrency_results = results['collection_sizes'][size]['index_types'][idx_type]['concurrent_search']
                x_values = [r['num_concurrent'] for r in concurrency_results]
                y_values = [r['searches_per_second'] for r in concurrency_results]
                plt.plot(x_values, y_values, marker='o', label=f"{idx_type} ({size} vectors)")
        
        plt.title('Search Throughput vs Concurrency')
        plt.xlabel('Number of Concurrent Threads')
        plt.ylabel('Searches per Second (higher is better)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')  # Log scale for x-axis to better show the range
        
        # Plot 2: Average search time vs Concurrency level
        plt.subplot(2, 2, 2)
        for idx_type in index_types:
            for size in collection_sizes:
                concurrency_results = results['collection_sizes'][size]['index_types'][idx_type]['concurrent_search']
                x_values = [r['num_concurrent'] for r in concurrency_results]
                y_values = [r['avg_search_time'] * 1000 for r in concurrency_results]  # Convert to ms
                plt.plot(x_values, y_values, marker='o', label=f"{idx_type} ({size} vectors)")
        
        plt.title('Average Search Time vs Concurrency')
        plt.xlabel('Number of Concurrent Threads')
        plt.ylabel('Average Search Time (ms) (lower is better)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        # Plot 3: P95 search time vs Concurrency level
        plt.subplot(2, 2, 3)
        for idx_type in index_types:
            for size in collection_sizes:
                concurrency_results = results['collection_sizes'][size]['index_types'][idx_type]['concurrent_search']
                x_values = [r['num_concurrent'] for r in concurrency_results]
                y_values = [r['p95_search_time'] * 1000 for r in concurrency_results]  # Convert to ms
                plt.plot(x_values, y_values, marker='o', label=f"{idx_type} ({size} vectors)")
        
        plt.title('P95 Search Time vs Concurrency')
        plt.xlabel('Number of Concurrent Threads')
        plt.ylabel('P95 Search Time (ms) (lower is better)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        # Plot 4: Throughput efficiency (searches per second / concurrency)
        plt.subplot(2, 2, 4)
        for idx_type in index_types:
            for size in collection_sizes:
                concurrency_results = results['collection_sizes'][size]['index_types'][idx_type]['concurrent_search']
                x_values = [r['num_concurrent'] for r in concurrency_results]
                # Calculate efficiency: searches per second divided by number of concurrent threads
                # This shows how efficiently the system scales with more threads
                y_values = [r['searches_per_second'] / r['num_concurrent'] for r in concurrency_results]
                plt.plot(x_values, y_values, marker='o', label=f"{idx_type} ({size} vectors)")
        
        plt.title('Throughput Efficiency vs Concurrency')
        plt.xlabel('Number of Concurrent Threads')
        plt.ylabel('Searches/Second per Thread (higher is better)')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig('unified_benchmark_concurrency.png', bbox_inches='tight')
        
        # Create a third figure for recall and latency metrics
        plt.figure(figsize=(20, 15))
        
        # Plot 1: Recall rates (if available)
        plt.subplot(2, 2, 1)
        
        # Check if recall data is available
        has_recall = False
        for size in collection_sizes:
            for idx_type in index_types:
                if results['collection_sizes'][size]['index_types'][idx_type].get('recall'):
                    has_recall = True
                    break
            if has_recall:
                break
        
        if has_recall:
            # Find a size that has recall data
            recall_size = None
            for size in collection_sizes:
                if results['collection_sizes'][size]['index_types'][index_types[0]].get('recall'):
                    recall_size = size
                    break
            
            for idx_type in index_types:
                if not results['collection_sizes'][recall_size]['index_types'][idx_type].get('recall'):
                    continue
                    
                recall_data = results['collection_sizes'][recall_size]['index_types'][idx_type]['recall']['recall_results']
                x_labels = [str(r['parameters']) for r in recall_data]
                recall_values = [r['avg_recall'] * 100 for r in recall_data]
                plt.plot(x_labels, recall_values, marker='o', label=idx_type)
            
            plt.title(f'Recall Rates (Collection Size: {recall_size})')
            plt.xlabel('Search Parameters')
            plt.ylabel('Average Recall (%)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Recall data not available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=plt.gca().transAxes)
        
        # Plot 2: Latency percentiles for each index type
        plt.subplot(2, 2, 2)
        bar_width = 0.2
        x = np.arange(len(index_types))
        
        # Use the largest collection size for latency comparison
        largest_size = max(collection_sizes)
        
        p50_values = [results['collection_sizes'][largest_size]['index_types'][idx_type]['latency_distribution']['latency_percentiles_ms']['p50'] for idx_type in index_types]
        p95_values = [results['collection_sizes'][largest_size]['index_types'][idx_type]['latency_distribution']['latency_percentiles_ms']['p95'] for idx_type in index_types]
        p99_values = [results['collection_sizes'][largest_size]['index_types'][idx_type]['latency_distribution']['latency_percentiles_ms']['p99'] for idx_type in index_types]
        
        plt.bar(x - bar_width, p50_values, width=bar_width, label='p50')
        plt.bar(x, p95_values, width=bar_width, label='p95')
        plt.bar(x + bar_width, p99_values, width=bar_width, label='p99')
        
        plt.title(f'Search Latency Percentiles (Collection Size: {largest_size})')
        plt.xlabel('Index Type')
        plt.ylabel('Latency (ms)')
        plt.xticks(x, index_types)
        plt.legend()
        plt.grid(True, axis='y')
        
        # Plot 3: Average position in results
        plt.subplot(2, 2, 3)
        for idx_type in index_types:
            positions_by_size = []
            for size in collection_sizes:
                search_results = results['collection_sizes'][size]['index_types'][idx_type]['search']['parameter_results']
                # Use the best parameter (usually the one with highest nprobe/ef)
                best_param = search_results[-1]
                positions_by_size.append(best_param['avg_position_when_found'])
            plt.plot(collection_sizes, positions_by_size, marker='o', label=idx_type)
        
        plt.title('Average Position of Target in Results (Best Parameter)')
        plt.xlabel('Collection Size (vectors)')
        plt.ylabel('Average Position (lower is better)')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Search time vs Collection size
        plt.subplot(2, 2, 4)
        for idx_type in index_types:
            times_by_size = []
            for size in collection_sizes:
                search_results = results['collection_sizes'][size]['index_types'][idx_type]['search']['parameter_results']
                # Use the best parameter (usually the one with highest nprobe/ef)
                best_param = search_results[-1]
                times_by_size.append(best_param['avg_time_per_search'] * 1000)  # Convert to ms
            plt.plot(collection_sizes, times_by_size, marker='o', label=idx_type)
        
        plt.title('Search Time vs Collection Size (Best Parameter)')
        plt.xlabel('Collection Size (vectors)')
        plt.ylabel('Average Search Time (ms)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('unified_benchmark_recall_latency.png', bbox_inches='tight')
        plt.close('all')

    def combine_benchmark_results(self, output_file='unified_benchmark_results.json'):
        """
        Combine individual benchmark result files into a complete result set
        
        This is useful for regenerating plots from existing result files
        without running any benchmarks.
        
        Args:
            output_file: Path to save the combined results
            
        Returns:
            Dictionary with combined benchmark results
        """
        print("Combining benchmark results...")
        
        # Initialize results structure
        results = {
            'collection_sizes': {},
            'parameters': {
                'dimensions': 1536,
                'model_name': "OpenAI (unified benchmark)",
            },
            'estimated_sizes': {}
        }
        
        # Find all individual result files
        result_files = [f for f in os.listdir('.') if f.startswith('unified_benchmark_') and f.endswith('.json') and '_' in f]
        
        if not result_files:
            print("No result files found.")
            return results
        
        # Extract collection sizes and index types from filenames
        sizes = set()
        index_types = set()
        concurrent_levels = set()
        
        # Process each file
        for file in result_files:
            try:
                # Extract size and index type from filename
                # Format: unified_benchmark_SIZE_INDEXTYPE.json
                parts = file.replace('.json', '').split('_')
                if len(parts) >= 3:
                    size = parts[2]
                    index_type = parts[3]
                    
                    # Load the file
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    # Initialize size entry if needed
                    if size not in results['collection_sizes']:
                        results['collection_sizes'][size] = {'index_types': {}}
                    
                    # Add the results
                    results['collection_sizes'][size]['index_types'][index_type] = data
                    
                    # Track sizes and index types
                    sizes.add(int(size))
                    index_types.add(index_type)
                    
                    # Track concurrent levels if present
                    if 'concurrent_search' in data:
                        for level in data['concurrent_search']:
                            if 'num_concurrent' in level:
                                concurrent_levels.add(level['num_concurrent'])
                    
                    # Add estimated sizes if present
                    if 'estimated_size_mb' in data:
                        if size not in results['estimated_sizes']:
                            results['estimated_sizes'][size] = {
                                'raw_data_size_mb': data['estimated_size_mb'],
                                'index_sizes_mb': {}
                            }
                        results['estimated_sizes'][size]['index_sizes_mb'][index_type] = data['estimated_index_size_mb']
                    
                    print(f"Added results for size {size}, index {index_type}")
                
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        
        # Update parameters
        results['parameters']['concurrent_levels_tested'] = sorted(list(concurrent_levels))
        
        # Save combined results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Combined results saved to {output_file}")
        print(f"Found {len(sizes)} collection sizes: {sorted(list(sizes))}")
        print(f"Found {len(index_types)} index types: {sorted(list(index_types))}")
        
        # Generate plots
        self.plot_unified_results(results, sorted(list(concurrent_levels)))
        
        return results

    def run_cpu_gpu_comparison(self, collection_sizes=[1000, 10000], 
                              concurrent_levels=[1, 10, 50, 100],
                              dim=1536, 
                              target_text="This is a specific text that we want to find later",
                              resume=True):
        """
        Run a comparative benchmark between CPU and GPU indexes
        
        This benchmark runs both CPU and GPU tests and generates comparative visualizations.
        
        Args:
            collection_sizes: List of collection sizes to test
            concurrent_levels: List of concurrent thread counts to test
            dim: Vector dimension
            target_text: Text to search for
            resume: Whether to resume from previous runs
            
        Returns:
            Dictionary with comparative benchmark results
        """
        print("\n=== Running CPU/GPU Comparison Benchmark ===\n")
        
        # Run CPU benchmark if needed
        cpu_results_file = "unified_benchmark_results_cpu.json"
        if resume and os.path.exists(cpu_results_file):
            print(f"Loading existing CPU benchmark results from {cpu_results_file}")
            with open(cpu_results_file, 'r') as f:
                cpu_results = json.load(f)
        else:
            print("Running CPU benchmark...")
            cpu_results = self.run_unified_benchmark(
                collection_sizes=collection_sizes,
                concurrent_levels=concurrent_levels,
                dim=dim,
                target_text=target_text,
                resume=resume,
                use_gpu=False
            )
        
        # Run GPU benchmark if needed
        gpu_results_file = "unified_benchmark_results_gpu.json"
        if resume and os.path.exists(gpu_results_file):
            print(f"Loading existing GPU benchmark results from {gpu_results_file}")
            with open(gpu_results_file, 'r') as f:
                gpu_results = json.load(f)
        else:
            print("Running GPU benchmark...")
            gpu_results = self.run_unified_benchmark(
                collection_sizes=collection_sizes,
                concurrent_levels=concurrent_levels,
                dim=dim,
                target_text=target_text,
                resume=resume,
                use_gpu=True
            )
        
        # Generate comparative visualizations
        print("Generating comparative visualizations...")
        self.plot_cpu_gpu_comparison(cpu_results, gpu_results, collection_sizes, concurrent_levels)
        
        # Return both result sets
        return {
            'cpu_results': cpu_results,
            'gpu_results': gpu_results
        }

    def plot_cpu_gpu_comparison(self, cpu_results, gpu_results, collection_sizes, concurrent_levels):
        """
        Generate comparative visualizations between CPU and GPU indexes
        
        Args:
            cpu_results: Results from CPU benchmark
            gpu_results: Results from GPU benchmark
            collection_sizes: List of collection sizes tested
            concurrent_levels: List of concurrent thread counts tested
        """
        # Convert collection sizes to strings for dictionary lookup
        collection_sizes_str = [str(size) for size in collection_sizes]
        
        # Get CPU and GPU index types
        cpu_index_types = list(self.index_configs.keys())
        gpu_index_types = list(self.gpu_index_configs.keys())
        
        # Create a figure for index creation time comparison
        plt.figure(figsize=(15, 10))
        plt.title('Index Creation Time: CPU vs GPU')
        
        # Set up bar positions
        bar_width = 0.15
        index = np.arange(len(collection_sizes))
        
        # Plot CPU index creation times
        for i, idx_type in enumerate(cpu_index_types):
            times = []
            for size_str in collection_sizes_str:
                if size_str in cpu_results['collection_sizes'] and idx_type in cpu_results['collection_sizes'][size_str]['index_types']:
                    times.append(cpu_results['collection_sizes'][size_str]['index_types'][idx_type]['insertion']['index_time_after_insert'])
                else:
                    times.append(0)
            plt.bar(index + i*bar_width, times, bar_width, label=f"CPU {idx_type}")
        
        # Plot GPU index creation times
        for i, idx_type in enumerate(gpu_index_types):
            times = []
            for size_str in collection_sizes_str:
                if size_str in gpu_results['collection_sizes'] and idx_type in gpu_results['collection_sizes'][size_str]['index_types']:
                    times.append(gpu_results['collection_sizes'][size_str]['index_types'][idx_type]['insertion']['index_time_after_insert'])
                else:
                    times.append(0)
            plt.bar(index + (i+len(cpu_index_types))*bar_width, times, bar_width, label=f"GPU {idx_type}")
        
        plt.xlabel('Collection Size')
        plt.ylabel('Index Creation Time (seconds)')
        plt.xticks(index + bar_width * (len(cpu_index_types) + len(gpu_index_types) - 1) / 2, collection_sizes)
        plt.legend()
        plt.grid(axis='y')
        plt.savefig('cpu_gpu_index_creation_time.png', bbox_inches='tight')
        plt.close()
        
        # Create a figure for search throughput comparison
        plt.figure(figsize=(15, 10))
        plt.title('Search Throughput: CPU vs GPU (Best Parameters)')
        
        # Use the largest collection size for comparison
        largest_size = max(collection_sizes_str)
        
        # Set up bar positions
        index = np.arange(len(concurrent_levels))
        
        # Plot CPU search throughput
        for i, idx_type in enumerate(cpu_index_types):
            if largest_size in cpu_results['collection_sizes'] and idx_type in cpu_results['collection_sizes'][largest_size]['index_types']:
                throughputs = []
                for level in concurrent_levels:
                    # Find the concurrent result for this level
                    for result in cpu_results['collection_sizes'][largest_size]['index_types'][idx_type]['concurrent_search']:
                        if result['num_concurrent'] == level:
                            throughputs.append(result['searches_per_second'])
                            break
                    else:
                        throughputs.append(0)
                plt.bar(index + i*bar_width, throughputs, bar_width, label=f"CPU {idx_type}")
        
        # Plot GPU search throughput
        for i, idx_type in enumerate(gpu_index_types):
            if largest_size in gpu_results['collection_sizes'] and idx_type in gpu_results['collection_sizes'][largest_size]['index_types']:
                throughputs = []
                for level in concurrent_levels:
                    # Find the concurrent result for this level
                    for result in gpu_results['collection_sizes'][largest_size]['index_types'][idx_type]['concurrent_search']:
                        if result['num_concurrent'] == level:
                            throughputs.append(result['searches_per_second'])
                            break
                    else:
                        throughputs.append(0)
                plt.bar(index + (i+len(cpu_index_types))*bar_width, throughputs, bar_width, label=f"GPU {idx_type}")
        
        plt.xlabel('Concurrent Threads')
        plt.ylabel('Searches per Second')
        plt.xticks(index + bar_width * (len(cpu_index_types) + len(gpu_index_types) - 1) / 2, concurrent_levels)
        plt.legend()
        plt.grid(axis='y')
        plt.savefig('cpu_gpu_search_throughput.png', bbox_inches='tight')
        plt.close()
        
        # Create a figure for search latency comparison
        plt.figure(figsize=(15, 10))
        plt.title('Search Latency (P95): CPU vs GPU')
        
        # Set up bar positions
        index = np.arange(len(concurrent_levels))
        
        # Plot CPU search latency
        for i, idx_type in enumerate(cpu_index_types):
            if largest_size in cpu_results['collection_sizes'] and idx_type in cpu_results['collection_sizes'][largest_size]['index_types']:
                latencies = []
                for level in concurrent_levels:
                    # Find the concurrent result for this level
                    for result in cpu_results['collection_sizes'][largest_size]['index_types'][idx_type]['concurrent_search']:
                        if result['num_concurrent'] == level:
                            latencies.append(result['p95_search_time'] * 1000)  # Convert to ms
                            break
                    else:
                        latencies.append(0)
                plt.bar(index + i*bar_width, latencies, bar_width, label=f"CPU {idx_type}")
        
        # Plot GPU search latency
        for i, idx_type in enumerate(gpu_index_types):
            if largest_size in gpu_results['collection_sizes'] and idx_type in gpu_results['collection_sizes'][largest_size]['index_types']:
                latencies = []
                for level in concurrent_levels:
                    # Find the concurrent result for this level
                    for result in gpu_results['collection_sizes'][largest_size]['index_types'][idx_type]['concurrent_search']:
                        if result['num_concurrent'] == level:
                            latencies.append(result['p95_search_time'] * 1000)  # Convert to ms
                            break
                    else:
                        latencies.append(0)
                plt.bar(index + (i+len(cpu_index_types))*bar_width, latencies, bar_width, label=f"GPU {idx_type}")
        
        plt.xlabel('Concurrent Threads')
        plt.ylabel('P95 Search Latency (ms)')
        plt.xticks(index + bar_width * (len(cpu_index_types) + len(gpu_index_types) - 1) / 2, concurrent_levels)
        plt.legend()
        plt.grid(axis='y')
        plt.savefig('cpu_gpu_search_latency.png', bbox_inches='tight')
        plt.close()
        
        # Create a figure for memory usage comparison
        plt.figure(figsize=(15, 10))
        plt.title('Memory Usage: CPU vs GPU')
        
        # Set up bar positions
        index = np.arange(len(collection_sizes))
        
        # Plot CPU memory usage
        for i, idx_type in enumerate(cpu_index_types):
            memory = []
            for size_str in collection_sizes_str:
                if size_str in cpu_results['collection_sizes'] and idx_type in cpu_results['collection_sizes'][size_str]['index_types']:
                    memory.append(cpu_results['collection_sizes'][size_str]['index_types'][idx_type]['memory']['total_memory_mb'])
                else:
                    memory.append(0)
            plt.bar(index + i*bar_width, memory, bar_width, label=f"CPU {idx_type}")
        
        # Plot GPU memory usage
        for i, idx_type in enumerate(gpu_index_types):
            memory = []
            for size_str in collection_sizes_str:
                if size_str in gpu_results['collection_sizes'] and idx_type in gpu_results['collection_sizes'][size_str]['index_types']:
                    memory.append(gpu_results['collection_sizes'][size_str]['index_types'][idx_type]['memory']['total_memory_mb'])
                else:
                    memory.append(0)
            plt.bar(index + (i+len(cpu_index_types))*bar_width, memory, bar_width, label=f"GPU {idx_type}")
        
        plt.xlabel('Collection Size')
        plt.ylabel('Memory Usage (MB)')
        plt.xticks(index + bar_width * (len(cpu_index_types) + len(gpu_index_types) - 1) / 2, collection_sizes)
        plt.legend()
        plt.grid(axis='y')
        plt.savefig('cpu_gpu_memory_usage.png', bbox_inches='tight')
        plt.close()
        
        # Create a figure for recall comparison
        plt.figure(figsize=(15, 10))
        plt.title('Recall Rate: CPU vs GPU (Best Parameters)')
        
        # Set up bar positions
        index = np.arange(1)
        
        # Plot CPU recall
        cpu_recalls = []
        cpu_labels = []
        for idx_type in cpu_index_types:
            if largest_size in cpu_results['collection_sizes'] and idx_type in cpu_results['collection_sizes'][largest_size]['index_types']:
                recall_data = cpu_results['collection_sizes'][largest_size]['index_types'][idx_type].get('recall')
                if recall_data and recall_data['recall_results']:
                    # Use the best parameter (usually the last one)
                    best_recall = recall_data['recall_results'][-1]['avg_recall'] * 100
                    cpu_recalls.append(best_recall)
                    cpu_labels.append(f"CPU {idx_type}")
        
        # Plot GPU recall
        gpu_recalls = []
        gpu_labels = []
        for idx_type in gpu_index_types:
            if largest_size in gpu_results['collection_sizes'] and idx_type in gpu_results['collection_sizes'][largest_size]['index_types']:
                recall_data = gpu_results['collection_sizes'][largest_size]['index_types'][idx_type].get('recall')
                if recall_data and recall_data['recall_results']:
                    # Use the best parameter (usually the last one)
                    best_recall = recall_data['recall_results'][-1]['avg_recall'] * 100
                    gpu_recalls.append(best_recall)
                    gpu_labels.append(f"GPU {idx_type}")
        
        # Combine all recalls and labels
        all_recalls = cpu_recalls + gpu_recalls
        all_labels = cpu_labels + gpu_labels
        
        # Create a horizontal bar chart
        plt.figure(figsize=(15, 10))
        plt.barh(all_labels, all_recalls)
        plt.xlabel('Recall Rate (%)')
        plt.title('Recall Rate: CPU vs GPU (Best Parameters)')
        plt.grid(axis='x')
        plt.savefig('cpu_gpu_recall.png', bbox_inches='tight')
        plt.close()

def generate_fake_vectors(num_vectors):
    """Generate fake vectors and texts for testing"""
    # Generate random vectors with dimension 1536
    vectors = np.random.rand(num_vectors, 1536).astype(np.float32)
    # Generate fake texts
    texts = [fake.text(max_nb_chars=200) for _ in range(num_vectors)]
    return vectors, texts

def get_embedding(text):
    """Get real embedding from OpenAI API"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def setup_milvus_collection(collection_name="benchmark_collection"):
    """Setup Milvus collection with required schema"""
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")
    
    # Define collection schema
    dim = 1536
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description="Benchmark collection")
    
    # Create collection
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection

def benchmark_insertion(collection_name, batch_sizes=[1000, 5000, 10000],
                      total_vectors=1000000, dim=1536, index_type="IVF_FLAT"):
    results = []
   
    # Get the appropriate index configuration
    if index_type in self.index_configs:
        index_config = self.index_configs[index_type]
    elif index_type in self.gpu_index_configs:
        index_config = self.gpu_index_configs[index_type]
    else:
        raise ValueError(f"Unknown index type: {index_type}")
   
    for batch_size in batch_sizes:
        # Generate random vectors instead of using embeddings
        texts = []
        embeddings = []
        for i in range(0, total_vectors, batch_size):
            current_batch_size = min(batch_size, total_vectors - i)
            
            # Generate random texts and vectors
            batch_texts = [self.generate_random_text() for _ in range(current_batch_size)]
            batch_embeddings = np.random.rand(current_batch_size, dim).astype(np.float32)
            
            texts.extend(batch_texts)
            embeddings.extend(batch_embeddings)
        
        # Insert data
        insert_start_time = time.time()
        
        # Prepare data for insertion
        data = []
        for i in range(len(texts)):
            data.append({
                "id": i,
                "text": texts[i],
                "embedding": embeddings[i].tolist()
            })
        
        # Insert in batches
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            self.client.insert(
                collection_name=collection_name,
                data=batch_data
            )
        
        insert_time = time.time() - insert_start_time
        
        # Create index after insertion
        index_start_time = time.time()
        
        # Prepare index parameters
        index_params = self.client.prepare_index_params()
        
        # Add index for vector field
        index_params.add_index(
            field_name="embedding",
            index_type=index_config["index_type"],
            metric_type=index_config["metric_type"],
            params=index_config["params"]
        )
        
        # Create index
        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )
        
        index_time_after = time.time() - index_start_time
       
        results.append({
            'batch_size': batch_size,
            'insert_time': insert_time,
            'vectors_per_second': total_vectors / insert_time,
            'index_time_after_insert': index_time_after
        })
       
    return results

def benchmark_search(collection_name, target_embedding, target_id,
                    num_searches=100, topk=100):
    # Load collection
    self.client.load_collection(collection_name)
    
    # Get the index type from collection
    index_info = self.client.describe_index(
        collection_name=collection_name,
        index_name="embedding"  # Index name is the same as field name by default
    )
    index_type = index_info['index_type']
    
    # Define search parameters for different index types
    search_params = {
        # CPU indexes
        'IVF_FLAT': [
            {"nprobe": 10},
            {"nprobe": 50},
            {"nprobe": 100},
            {"nprobe": 500}
        ],
        'IVF_SQ8': [
            {"nprobe": 10},
            {"nprobe": 50},
            {"nprobe": 100},
            {"nprobe": 500}
        ],
        'HNSW': [
            {"ef": 10},
            {"ef": 50},
            {"ef": 100},
            {"ef": 500}
        ],
        # GPU indexes
        'GPU_CAGRA': [
            {"itopk_size": 64, "search_width": 1},
            {"itopk_size": 128, "search_width": 2},
            {"itopk_size": 256, "search_width": 4},
            {"itopk_size": 512, "search_width": 8}
        ],
        'GPU_IVF_FLAT': [
            {"nprobe": 10},
            {"nprobe": 50},
            {"nprobe": 100},
            {"nprobe": 500}
        ],
        'GPU_IVF_PQ': [
            {"nprobe": 10},
            {"nprobe": 50},
            {"nprobe": 100},
            {"nprobe": 500}
        ],
        'GPU_BRUTE_FORCE': [
            {}  # No parameters needed
        ]
    }
    
    results = {
        'num_searches': num_searches,
        'topk': topk,
        'parameter_results': []
    }
    
    # Test each search parameter
    for search_param in search_params.get(index_type, [{"nprobe": 10}]):
        # Warm-up
        self.client.search(
            collection_name=collection_name,
            data=[target_embedding],
            anns_field="embedding",
            param={
                "metric_type": "L2",
                "params": search_param
            },
            limit=topk,
            output_fields=["text"]
        )
        
        start_time = time.time()
        found_count = 0
        positions = []  # Track positions when found
        
        for _ in range(num_searches):
            search_results = self.client.search(
                collection_name=collection_name,
                data=[target_embedding],
                anns_field="embedding",
                param={
                    "metric_type": "L2",
                    "params": search_param
                },
                limit=topk,
                output_fields=["text"]
            )
            
            # Find position of target_id in results
            found = False
            for pos, hit in enumerate(search_results[0]):
                if hit["id"] == target_id:  # Adjust based on actual result format
                    found = True
                    found_count += 1
                    positions.append(pos + 1)  # Add 1 for 1-based position
                    break
            if not found:
                positions.append(None)  # Mark as not found
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate average position when found
        found_positions = [p for p in positions if p is not None]
        avg_position = sum(found_positions) / len(found_positions) if found_positions else None
        
        param_result = {
            'parameters': search_param,
            'total_time': total_time,
            'searches_per_second': num_searches / total_time,
            'avg_time_per_search': total_time / num_searches,
            'target_found_rate': found_count / num_searches,
            'avg_position_when_found': avg_position,
            'position_stats': {
                'min': min(found_positions) if found_positions else None,
                'max': max(found_positions) if found_positions else None,
                'positions': positions
            }
        }
        results['parameter_results'].append(param_result)
    
    return results

def main():
    # Initialize the benchmark class
    benchmark = VectorBenchmark()
    
    # Choose which benchmarks to run
    run_cpu = False
    run_gpu = False
    run_comparison = True
    combine_only = False  # Set to True to just combine existing results without running benchmarks
    resume = True  # Set to True to resume from previous runs
    
    if combine_only:
        # Just combine existing results and generate plots
        print("\n=== Combining Existing Benchmark Results ===\n")
        combined_results_cpu = benchmark.combine_benchmark_results(output_file="unified_benchmark_results_cpu.json")
        combined_results_gpu = benchmark.combine_benchmark_results(output_file="unified_benchmark_results_gpu.json")
    else:
        if run_cpu:
            # Run CPU benchmark
            print("\n=== Running CPU Unified Benchmark ===\n")
            unified_results_cpu = benchmark.run_unified_benchmark(
                collection_sizes=[100, 1000, 5000, 10000],  # Collection sizes to test
                concurrent_levels=[1, 10, 50, 100, 200, 500],  # Concurrency levels
                dim=1536,  # OpenAI embedding dimension
                target_text="This is a specific text that we want to find later",
                resume=resume,  # Whether to resume from previous runs
                use_gpu=False
            )
        
        if run_gpu:
            # Run GPU benchmark
            print("\n=== Running GPU Unified Benchmark ===\n")
            unified_results_gpu = benchmark.run_unified_benchmark(
                collection_sizes=[100, 1000, 5000, 10000],  # Collection sizes to test
                concurrent_levels=[1, 10, 50, 100, 200, 500],  # Concurrency levels
                dim=1536,  # OpenAI embedding dimension
                target_text="This is a specific text that we want to find later",
                resume=resume,  # Whether to resume from previous runs
                use_gpu=True
            )
        
        if run_comparison:
            # Run CPU/GPU comparison benchmark
            comparison_results = benchmark.run_cpu_gpu_comparison(
                collection_sizes=[1000, 10000],  # Use fewer sizes for comparison
                concurrent_levels=[1, 10, 50, 100],  # Use fewer concurrency levels
                dim=1536,
                target_text="This is a specific text that we want to find later",
                resume=resume
            )
    
    print("\nBenchmarks completed. Check the output files for results.")

if __name__ == "__main__":
    main()