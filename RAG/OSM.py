#!/usr/bin/env python3
# osm_to_rag_pipeline.py - Convert OSM data to RAG documents

import osmium
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pyproj import Transformer
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from collections import defaultdict
import re

# Initialize coordinate transformer
wgs84_to_sweref99 = Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)

@dataclass
class LocationDocument:
    """Structured document for RAG storage"""
    id: str
    name: str
    text: str
    location_type: str
    epsg3006_coords: Dict[str, float]
    metadata: Dict[str, Any]
    
class OSMHandler(osmium.SimpleHandler):
    """Handler to extract relevant features from OSM data"""
    
    def __init__(self):
        super().__init__()
        self.locations = []
        self.processed_count = 0
        
        # Define what we want to extract
        self.relevant_tags = {
            'place': ['city', 'town', 'village', 'hamlet', 'suburb', 'neighbourhood'],
            'natural': ['peak', 'water', 'wood', 'wetland', 'coastline'],
            'landuse': ['residential', 'industrial', 'commercial', 'forest'],
            'building': ['yes', 'residential', 'commercial', 'industrial'],
            'amenity': ['hospital', 'school', 'university', 'railway_station'],
            'tourism': ['attraction', 'museum', 'viewpoint'],
            'historic': ['castle', 'monument', 'archaeological_site']
        }
        
    def create_description(self, tags: Dict[str, str], location_type: str) -> str:
        """Create a natural language description from OSM tags"""
        descriptions = []
        
        # Basic description
        name = tags.get('name', 'Unnamed location')
        descriptions.append(f"{name} is a {location_type}")
        
        # Add population if available
        if 'population' in tags:
            descriptions.append(f"with a population of {tags['population']}")
        
        # Add elevation for peaks
        if 'ele' in tags:
            descriptions.append(f"at {tags['ele']} meters elevation")
            
        # Add municipality/county info
        if 'addr:municipality' in tags:
            descriptions.append(f"in {tags['addr:municipality']} municipality")
        elif 'is_in:municipality' in tags:
            descriptions.append(f"in {tags['is_in:municipality']} municipality")
            
        # Add any description or note
        if 'description' in tags:
            descriptions.append(f". {tags['description']}")
        elif 'note' in tags:
            descriptions.append(f". {tags['note']}")
            
        return ' '.join(descriptions) + '.'
    
    def process_location(self, obj, location_type: str):
        """Process a location and convert to document"""
        tags = dict(obj.tags)
        
        # Skip if no name
        if 'name' not in tags and 'name:sv' not in tags:
            return
            
        name = tags.get('name:sv', tags.get('name', ''))
        
        # Get coordinates
        if hasattr(obj, 'location'):
            # Node
            lat, lon = obj.location.lat, obj.location.lon
            x, y = wgs84_to_sweref99.transform(lon, lat)
            coords = {'x': x, 'y': y}
            coord_text = f"at coordinates EPSG:3006 X={x:.0f} Y={y:.0f}"
        elif hasattr(obj, 'nodes'):
            # Way - calculate centroid
            lats, lons = [], []
            for node in obj.nodes:
                if hasattr(node, 'location'):
                    lats.append(node.location.lat)
                    lons.append(node.location.lon)
            if lats and lons:
                lat, lon = np.mean(lats), np.mean(lons)
                x, y = wgs84_to_sweref99.transform(lon, lat)
                coords = {'x': x, 'y': y}
                coord_text = f"centered at EPSG:3006 X={x:.0f} Y={y:.0f}"
            else:
                return
        else:
            return
            
        # Create description
        description = self.create_description(tags, location_type)
        
        # Create searchable text
        search_text = f"{name} {description} Located {coord_text}. "
        
        # Add alternative names
        alt_names = []
        for key, value in tags.items():
            if key.startswith('name:') and key != 'name:sv':
                alt_names.append(value)
        if alt_names:
            search_text += f"Also known as: {', '.join(alt_names)}. "
            
        # Create metadata
        metadata = {
            'osm_id': obj.id,
            'type': location_type,
            'name': name,
            'tags': tags
        }
        
        # Add bounds for areas
        if hasattr(obj, 'bounds'):
            min_lon, min_lat = obj.bounds.min_lon, obj.bounds.min_lat
            max_lon, max_lat = obj.bounds.max_lon, obj.bounds.max_lat
            xmin, ymin = wgs84_to_sweref99.transform(min_lon, min_lat)
            xmax, ymax = wgs84_to_sweref99.transform(max_lon, max_lat)
            coords.update({
                'xmin': xmin, 'ymin': ymin,
                'xmax': xmax, 'ymax': ymax
            })
            metadata['has_bounds'] = True
            search_text += f"Bounding box: X={xmin:.0f}-{xmax:.0f} Y={ymin:.0f}-{ymax:.0f}. "
        
        # Create document
        doc = LocationDocument(
            id=f"{location_type}_{obj.id}",
            name=name,
            text=search_text,
            location_type=location_type,
            epsg3006_coords=coords,
            metadata=metadata
        )
        
        self.locations.append(doc)
        self.processed_count += 1
        
        if self.processed_count % 1000 == 0:
            print(f"Processed {self.processed_count} locations...")
    
    def node(self, n):
        """Process nodes (points)"""
        tags = n.tags
        
        # Check relevant tags
        for tag_type, values in self.relevant_tags.items():
            if tag_type in tags and tags[tag_type] in values:
                self.process_location(n, f"{tag_type}_{tags[tag_type]}")
                break
    
    def way(self, w):
        """Process ways (lines/areas)"""
        tags = w.tags
        
        # Check relevant tags
        for tag_type, values in self.relevant_tags.items():
            if tag_type in tags and tags[tag_type] in values:
                self.process_location(w, f"{tag_type}_{tags[tag_type]}")
                break
    
    def area(self, a):
        """Process areas (complex polygons)"""
        tags = a.tags
        
        # Check relevant tags
        for tag_type, values in self.relevant_tags.items():
            if tag_type in tags and tags[tag_type] in values:
                self.process_location(a, f"{tag_type}_{tags[tag_type]}")
                break

class SwedishOSMIndexer:
    """Index Swedish OSM data into ChromaDB for RAG"""
    
    def __init__(self, db_path: str = "./sweden_osm_rag"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Create collections for different types
        self.collections = {
            'cities': self.chroma_client.get_or_create_collection(
                name="swedish_cities",
                embedding_function=self.embedding_function
            ),
            'nature': self.chroma_client.get_or_create_collection(
                name="swedish_nature", 
                embedding_function=self.embedding_function
            ),
            'infrastructure': self.chroma_client.get_or_create_collection(
                name="swedish_infrastructure",
                embedding_function=self.embedding_function
            )
        }
        
    def categorize_location(self, location_type: str) -> str:
        """Categorize location into collection"""
        if any(x in location_type for x in ['city', 'town', 'village', 'suburb']):
            return 'cities'
        elif any(x in location_type for x in ['peak', 'water', 'wood', 'natural']):
            return 'nature'
        else:
            return 'infrastructure'
    
    def process_osm_file(self, osm_file_path: str, batch_size: int = 1000):
        """Process OSM file and index into ChromaDB"""
        print(f"Processing {osm_file_path}...")
        
        # Parse OSM file
        handler = OSMHandler()
        handler.apply_file(osm_file_path, locations=True)
        
        print(f"Extracted {len(handler.locations)} locations")
        
        # Group by category
        categorized = defaultdict(list)
        for loc in handler.locations:
            category = self.categorize_location(loc.location_type)
            categorized[category].append(loc)
        
        # Index in batches
        for category, locations in categorized.items():
            collection = self.collections[category]
            print(f"\nIndexing {len(locations)} {category}...")
            
            for i in range(0, len(locations), batch_size):
                batch = locations[i:i + batch_size]
                
                # Prepare batch data
                ids = [loc.id for loc in batch]
                documents = [loc.text for loc in batch]
                metadatas = []
                
                for loc in batch:
                    metadata = {
                        'name': loc.name,
                        'type': loc.location_type,
                        'osm_id': loc.metadata['osm_id']
                    }
                    
                    # Add coordinates
                    for key, value in loc.epsg3006_coords.items():
                        metadata[f'epsg3006_{key}'] = value
                    
                    # Add selected tags
                    tags = loc.metadata.get('tags', {})
                    if 'population' in tags:
                        metadata['population'] = int(tags['population']) if tags['population'].isdigit() else 0
                    if 'ele' in tags:
                        metadata['elevation'] = float(tags['ele']) if tags['ele'].replace('.', '').isdigit() else 0
                    
                    metadatas.append(metadata)
                
                # Add to collection
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                
                print(f"  Indexed batch {i//batch_size + 1}/{(len(locations) + batch_size - 1)//batch_size}")
        
        print("\nIndexing complete!")
        
    def search(self, query: str, category: Optional[str] = None, n_results: int = 5):
        """Search the indexed data"""
        if category and category in self.collections:
            collections = [self.collections[category]]
        else:
            collections = list(self.collections.values())
        
        all_results = []
        
        for collection in collections:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                all_results.append({
                    'text': doc,
                    'metadata': metadata,
                    'score': 1 - distance,
                    'collection': collection.name
                })
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        return all_results[:n_results]

# Example usage script
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Index Swedish OSM data for RAG")
    parser.add_argument("--osm-file", help="Path to OSM file (e.g., sweden-latest.osm.pbf)")
    parser.add_argument("--index", action="store_true", help="Index the OSM file")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--category", choices=['cities', 'nature', 'infrastructure'],
                        help="Limit search to category")
    
    args = parser.parse_args()
    
    indexer = SwedishOSMIndexer()
    
    if args.index and args.osm_file:
        indexer.process_osm_file(args.osm_file)
    
    if args.search:
        results = indexer.search(args.search, args.category)
        print(f"\nSearch results for '{args.search}':")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['metadata']['name']} ({result['collection']})")
            print(f"   Score: {result['score']:.3f}")
            print(f"   {result['text'][:200]}...")
            if 'epsg3006_x' in result['metadata']:
                print(f"   Coordinates: X={result['metadata']['epsg3006_x']:.0f}, Y={result['metadata']['epsg3006_y']:.0f}")

if __name__ == "__main__":
    main()