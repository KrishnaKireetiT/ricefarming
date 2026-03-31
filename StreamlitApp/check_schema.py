#!/usr/bin/env python3
"""Quick script to check actual Section node properties"""
from neo4j import GraphDatabase

NEO4J_URI = "neo4j+s://2d074b24.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cnSN_ZMqaui2WG40oeiQZwXYoG4fBoVYE9z7-woOAsI"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

with driver.session(database="neo4j") as session:
    # Check Section properties
    result = session.run("MATCH (n:Section) RETURN properties(n) as props LIMIT 1")
    record = result.single()
    if record:
        print("Section node properties:")
        print(list(record['props'].keys()))
        print("\nFull sample:")
        print(record['props'])
    
    # Count sections
    count = session.run("MATCH (n:Section) RETURN count(n) as count").single()['count']
    print(f"\nTotal Section nodes: {count}")

driver.close()
