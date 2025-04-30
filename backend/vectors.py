import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import time
from tqdm import tqdm
import json
import random
import string
from pymilvus import connections, Collection, DataType, FieldSchema, CollectionSchema, utility
import matplotlib.pyplot as plt
import struct

class VectorBenchmark:
    def __init__(self, host='localhost', port='19530', device='cuda'):
        self.host = host
        self.port = port
        self.device = 'cpu' if not torch.cuda.is_available() and device == 'cuda' else device
       
        # Initialize embedding model
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
       
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
        self.connect()

    def connect(self):
        connections.connect(host=self.host, port=self.port)

    def get_embedding(self, texts):
        """Generate embeddings using GPU"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
       
        with torch.no_grad():
            outputs = self.model(**inputs)
       
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

    def generate_random_text(self, length=100):
        return ''.join(random.choices(string.ascii_letters + ' ', k=length))

    def create_collection(self, dim=384, collection_name="benchmark_collection", index_type="IVF_FLAT", create_index=True):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
           
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields=fields)
        collection = Collection(name=collection_name, schema=schema)
       
        if create_index:
            start_time = time.time()
            collection.create_index("embedding", self.index_configs[index_type])
            index_time = time.time() - start_time
            return collection, index_time
        return collection, 0

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

    def benchmark_insertion(self, collection, batch_sizes=[1000, 5000, 10000],
                          total_vectors=1000000, dim=384, index_type="IVF_FLAT"):
        results = []
       
        for batch_size in batch_sizes:
            # Generate data in batches
            texts = []
            embeddings = []
            for i in range(0, total_vectors, batch_size):
                batch_texts = [self.generate_random_text() for _ in range(min(batch_size, total_vectors - i))]
                batch_embeddings = self.get_embedding(batch_texts)
                texts.extend(batch_texts)
                embeddings.extend(batch_embeddings)

            collection.drop()
            collection, _ = self.create_collection(dim=dim, create_index=False)
           
            # Insert data
            insert_start_time = time.time()
            for i in tqdm(range(0, total_vectors, batch_size)):
                end_idx = min(i + batch_size, total_vectors)
                entities = [
                    list(range(i, end_idx)),
                    texts[i:end_idx],
                    embeddings[i:end_idx]
                ]
                collection.insert(entities)
           
            collection.flush()
            insert_time = time.time() - insert_start_time
           
            # Create index after insertion
            index_start_time = time.time()
            collection.create_index("embedding", self.index_configs[index_type])
            index_time_after = time.time() - index_start_time
           
            results.append({
                'batch_size': batch_size,
                'insert_time': insert_time,
                'vectors_per_second': total_vectors / insert_time,
                'index_time_after_insert': index_time_after
            })
           
        return results

    def benchmark_search(self, collection, target_embedding, target_id,
                        num_searches=100, topk=100):
        collection.load()
       
        # Warm-up
        collection.search(
            [target_embedding],
            "embedding",
            {"metric_type": "L2", "params": {"nprobe": 10}},
            topk,
            output_fields=["text"]
        )
       
        start_time = time.time()
       
        for _ in range(num_searches):
            results = collection.search(
                [target_embedding],
                "embedding",
                {"metric_type": "L2", "params": {"nprobe": 10}},
                topk,
                output_fields=["text"]
            )
           
            found = any(hit.id == target_id for hit in results[0])
       
        end_time = time.time()
       
        return {
            'num_searches': num_searches,
            'total_time': end_time - start_time,
            'searches_per_second': num_searches / (end_time - start_time),
            'target_found': found,
            'avg_time_per_search': (end_time - start_time) / num_searches
        }

    def plant_search_data(self, collection, target_text, target_embedding):
        collection.insert([
            [collection.num_entities],
            [target_text],
            [target_embedding]
        ])
        collection.flush()
        return collection.num_entities - 1

    def run_complete_benchmark(self, collection_sizes=[10000, 100000, 1000000],
                             dim=384, target_text="FINDME"):
        target_embedding = self.get_embedding([target_text])[0]
           
        results = {
            'collection_sizes': {},
            'parameters': {
                'dimensions': dim,
                'target_text': target_text,
                'model_name': self.model_name
            }
        }
       
        for size in collection_sizes:
            print(f"\nTesting collection size: {size}")
            size_results = {'index_types': {}}
           
            for index_type in self.index_configs.keys():
                print(f"\nTesting index: {index_type}")
                collection, index_time_empty = self.create_collection(dim=dim, index_type=index_type)
               
                insert_results = self.benchmark_insertion(
                    collection,
                    batch_sizes=[5000],
                    total_vectors=size,
                    dim=dim,
                    index_type=index_type
                )
               
                target_id = self.plant_search_data(collection, target_text, target_embedding)
                search_results = self.benchmark_search(collection, target_embedding, target_id)
               
                results_for_size = {
                    'insertion': insert_results[0],
                    'search': search_results,
                    'index_time_empty': index_time_empty
                }
                size_results['index_types'][index_type] = results_for_size
               
                collection.drop()
           
            results['collection_sizes'][size] = size_results
       
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=4)
       
        self.plot_results(results)
        return results

    def plot_results(self, results):
        collection_sizes = list(results['collection_sizes'].keys())
        index_types = list(results['collection_sizes'][collection_sizes[0]]['index_types'].keys())
       
        plt.figure(figsize=(20, 5))
       
        # Plot insertion speeds
        plt.subplot(1, 3, 1)
        for idx_type in index_types:
            speeds = [results['collection_sizes'][size]['index_types'][idx_type]['insertion']['vectors_per_second']
                     for size in collection_sizes]
            plt.plot(collection_sizes, speeds, marker='o', label=idx_type)
        plt.title('Insertion Performance')
        plt.xlabel('Collection Size')
        plt.ylabel('Vectors per Second')
        plt.legend()
        plt.xscale('log')
       
        # Plot search speeds
        plt.subplot(1, 3, 2)
        for idx_type in index_types:
            speeds = [results['collection_sizes'][size]['index_types'][idx_type]['search']['searches_per_second']
                     for size in collection_sizes]
            plt.plot(collection_sizes, speeds, marker='o', label=idx_type)
        plt.title('Search Performance')
        plt.xlabel('Collection Size')
        plt.ylabel('Searches per Second')
        plt.legend()
        plt.xscale('log')
       
        # Plot index creation times
        plt.subplot(1, 3, 3)
        for idx_type in index_types:
            times = [results['collection_sizes'][size]['index_types'][idx_type]['insertion']['index_time_after_insert']
                    for size in collection_sizes]
            plt.plot(collection_sizes, times, marker='o', label=idx_type)
        plt.title('Index Creation Time')
        plt.xlabel('Collection Size')
        plt.ylabel('Seconds')
        plt.legend()
        plt.xscale('log')
       
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        plt.close()

if __name__ == "__main__":
    benchmark = VectorBenchmark()
    results = benchmark.run_complete_benchmark(
        collection_sizes=[10000, 100000, 1000000],
        target_text="This is a specific text we want to find later"
    )