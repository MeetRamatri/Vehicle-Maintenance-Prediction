"""
Unit tests for the RAG Chunking module.
"""
import os
import json
import csv
from rag_pipeline.chunking import (
    load_instruction_chunks,
    load_enriched_chunks,
    get_all_chunks,
    MAINTENANCE_KNOWLEDGE,
)


class TestLoadInstructionChunks:
    """Tests for load_instruction_chunks function."""

    def test_load_from_valid_jsonl(self, temp_directory, sample_instruction_data):
        """Test loading chunks from a valid JSONL file."""
        jsonl_path = os.path.join(temp_directory, "test_instructions.jsonl")
        
        # Write test JSONL file
        with open(jsonl_path, "w") as f:
            for record in sample_instruction_data:
                f.write(json.dumps(record) + "\n")
        
        chunks = load_instruction_chunks(jsonl_path)
        
        assert len(chunks) == 2
        assert "high mileage vehicles" in chunks[0].lower()
        assert "brake pads" in chunks[1].lower()

    def test_load_from_nonexistent_file(self):
        """Test loading from a file that doesn't exist."""
        chunks = load_instruction_chunks("/nonexistent/path/file.jsonl")
        assert chunks == []

    def test_empty_jsonl_file(self, temp_directory):
        """Test loading from an empty JSONL file."""
        jsonl_path = os.path.join(temp_directory, "empty.jsonl")
        open(jsonl_path, "w").close()
        
        chunks = load_instruction_chunks(jsonl_path)
        assert chunks == []

    def test_chunk_format(self, temp_directory):
        """Test that chunks have the correct Q: A: format."""
        jsonl_path = os.path.join(temp_directory, "test.jsonl")
        record = {
            "instruction": "Test question?",
            "input": "",
            "output": "Test answer."
        }
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(record) + "\n")
        
        chunks = load_instruction_chunks(jsonl_path)
        
        assert len(chunks) == 1
        assert chunks[0].startswith("Q:")
        assert "A:" in chunks[0]

    def test_handles_blank_lines(self, temp_directory):
        """Test that blank lines are properly skipped."""
        jsonl_path = os.path.join(temp_directory, "with_blanks.jsonl")
        record = {"instruction": "Q", "input": "", "output": "A"}
        
        with open(jsonl_path, "w") as f:
            f.write("\n")
            f.write(json.dumps(record) + "\n")
            f.write("   \n")
            f.write("\n")
        
        chunks = load_instruction_chunks(jsonl_path)
        assert len(chunks) == 1


class TestLoadEnrichedChunks:
    """Tests for load_enriched_chunks function."""

    def test_load_from_valid_csv(self, temp_directory):
        """Test loading chunks from a valid CSV file."""
        csv_path = os.path.join(temp_directory, "test_enriched.csv")
        
        fieldnames = [
            "vehicle_summary", "maintenance_recommendation", "Vehicle_Model",
            "risk_level", "Mileage", "Vehicle_Age", "Tire_Condition",
            "Brake_Condition", "Battery_Status", "Reported_Issues"
        ]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                "vehicle_summary": "This is a test summary.",
                "maintenance_recommendation": "Service soon.",
                "Vehicle_Model": "Car",
                "risk_level": "High",
                "Mileage": "100000",
                "Vehicle_Age": "8",
                "Tire_Condition": "Worn Out",
                "Brake_Condition": "Good",
                "Battery_Status": "Weak",
                "Reported_Issues": "5",
            })
        
        chunks = load_enriched_chunks(csv_path)
        
        assert len(chunks) == 1
        assert "Car" in chunks[0]
        assert "100000" in chunks[0]
        assert "High" in chunks[0]

    def test_load_from_nonexistent_file(self):
        """Test loading from a file that doesn't exist."""
        chunks = load_enriched_chunks("/nonexistent/path/file.csv")
        assert chunks == []

    def test_max_rows_limit(self, temp_directory):
        """Test that max_rows parameter limits the number of chunks."""
        csv_path = os.path.join(temp_directory, "many_rows.csv")
        
        fieldnames = [
            "vehicle_summary", "maintenance_recommendation", "Vehicle_Model",
            "risk_level", "Mileage", "Vehicle_Age", "Tire_Condition",
            "Brake_Condition", "Battery_Status", "Reported_Issues"
        ]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(100):
                writer.writerow({
                    "vehicle_summary": f"Summary {i}",
                    "maintenance_recommendation": f"Recommendation {i}",
                    "Vehicle_Model": "Car",
                    "risk_level": "Low",
                    "Mileage": str(i * 1000),
                    "Vehicle_Age": "5",
                    "Tire_Condition": "Good",
                    "Brake_Condition": "Good",
                    "Battery_Status": "Good",
                    "Reported_Issues": "0",
                })
        
        chunks = load_enriched_chunks(csv_path, max_rows=10)
        assert len(chunks) == 10

    def test_default_max_rows(self, temp_directory):
        """Test that default max_rows is 500."""
        csv_path = os.path.join(temp_directory, "test.csv")
        
        fieldnames = ["vehicle_summary", "maintenance_recommendation", "Vehicle_Model",
                      "risk_level", "Mileage", "Vehicle_Age", "Tire_Condition",
                      "Brake_Condition", "Battery_Status", "Reported_Issues"]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(10):
                writer.writerow({
                    "vehicle_summary": f"S{i}", "maintenance_recommendation": f"R{i}",
                    "Vehicle_Model": "Car", "risk_level": "Low", "Mileage": "1000",
                    "Vehicle_Age": "1", "Tire_Condition": "Good", "Brake_Condition": "Good",
                    "Battery_Status": "Good", "Reported_Issues": "0",
                })
        
        chunks = load_enriched_chunks(csv_path)
        assert len(chunks) == 10  # Less than default max_rows of 500


class TestMaintenanceKnowledge:
    """Tests for the static MAINTENANCE_KNOWLEDGE constant."""

    def test_maintenance_knowledge_not_empty(self):
        """Test that MAINTENANCE_KNOWLEDGE has content."""
        assert len(MAINTENANCE_KNOWLEDGE) > 0

    def test_maintenance_knowledge_contains_basics(self):
        """Test that MAINTENANCE_KNOWLEDGE covers basic topics."""
        full_text = " ".join(MAINTENANCE_KNOWLEDGE).lower()
        
        # Check for key maintenance topics
        assert "oil" in full_text
        assert "tire" in full_text
        assert "brake" in full_text
        assert "battery" in full_text

    def test_all_items_are_strings(self):
        """Test that all items in MAINTENANCE_KNOWLEDGE are strings."""
        for item in MAINTENANCE_KNOWLEDGE:
            assert isinstance(item, str)
            assert len(item) > 0


class TestGetAllChunks:
    """Tests for get_all_chunks function."""

    def test_includes_maintenance_knowledge(self):
        """Test that get_all_chunks includes MAINTENANCE_KNOWLEDGE."""
        chunks = get_all_chunks()
        
        # Check that maintenance knowledge is included
        for knowledge in MAINTENANCE_KNOWLEDGE[:3]:  # Check first few
            assert knowledge in chunks

    def test_returns_list(self):
        """Test that get_all_chunks returns a list."""
        chunks = get_all_chunks()
        assert isinstance(chunks, list)

    def test_all_chunks_are_strings(self):
        """Test that all returned chunks are strings."""
        chunks = get_all_chunks()
        for chunk in chunks:
            assert isinstance(chunk, str)

    def test_no_empty_chunks(self):
        """Test that there are no empty chunks."""
        chunks = get_all_chunks()
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_custom_project_root(self, temp_directory):
        """Test get_all_chunks with a custom project root."""
        # Create empty llm_data directory
        llm_data_dir = os.path.join(temp_directory, "llm_data")
        os.makedirs(llm_data_dir, exist_ok=True)
        
        chunks = get_all_chunks(project_root=temp_directory)
        
        # Should still include MAINTENANCE_KNOWLEDGE even without files
        assert len(chunks) >= len(MAINTENANCE_KNOWLEDGE)
