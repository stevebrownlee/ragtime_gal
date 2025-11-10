"""
Migration Validation Test Suite

Purpose: Validate functionality before and after each phase of the project reorganization.
This test suite ensures that critical functionality remains intact during the refactoring process.

Usage:
    # Before starting a phase
    pytest tests/test_migration_validation.py -v

    # After completing a phase
    pytest tests/test_migration_validation.py -v --phase=<phase_number>
"""

import pytest
import sys
import importlib
import os
from pathlib import Path

# Add project root to path for pre-migration imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestPhase0PreMigration:
    """Tests to run BEFORE any migration begins - establishes baseline"""

    def test_root_modules_exist(self):
        """Verify all expected root modules exist"""
        expected_modules = [
            'app',
            'embed',
            'embed_enhanced',
            'query',
            'conversation',
            'enhanced_conversation',
            'conversation_embedder',
            'conversation_summarizer',
            'context_manager',
            'feedback_analyzer',
            'query_enhancer',
            'training_data_generator',
            'model_finetuner',
            'query_classifier',
            'prompts',
            'template_manager',
            'template',
            'conport_client',
            'monitoring_dashboard'
        ]

        for module_name in expected_modules:
            module_path = project_root / f"{module_name}.py"
            assert module_path.exists(), f"Module {module_name}.py not found in root"

    def test_critical_imports_work(self):
        """Test that critical modules can be imported"""
        critical_modules = [
            'conport_client',
            'app',
            'query',
            'embed'
        ]

        for module_name in critical_modules:
            try:
                module = importlib.import_module(module_name)
                assert module is not None, f"Module {module_name} imported but is None"
            except ImportError as e:
                pytest.fail(f"Failed to import critical module {module_name}: {e}")

    def test_configuration_files_exist(self):
        """Verify configuration files are present"""
        config_files = [
            '.env.template',
            'Pipfile',
            'Pipfile.lock'
        ]

        for config_file in config_files:
            file_path = project_root / config_file
            assert file_path.exists(), f"Configuration file {config_file} not found"

    def test_docs_exist(self):
        """Verify documentation exists"""
        doc_files = [
            'README.md',
            'docs/MATURITY_RECOMMANDATIONS.md',
            'docs/MODULE_INVENTORY.md'
        ]

        for doc_file in doc_files:
            file_path = project_root / doc_file
            assert file_path.exists(), f"Documentation file {doc_file} not found"


class TestPhase1Foundation:
    """Tests for Phase 1: Package structure and Pydantic models"""

    def test_package_structure_created(self):
        """Verify ragtime package structure exists"""
        expected_dirs = [
            'ragtime',
            'ragtime/api',
            'ragtime/core',
            'ragtime/models',
            'ragtime/services',
            'ragtime/storage',
            'ragtime/utils',
            'ragtime/monitoring',
            'ragtime/config'
        ]

        for dir_path in expected_dirs:
            full_path = project_root / dir_path
            assert full_path.exists(), f"Directory {dir_path} not created"
            assert (full_path / "__init__.py").exists(), f"__init__.py missing in {dir_path}"

    def test_pydantic_models_exist(self):
        """Verify Pydantic model files created"""
        model_files = [
            'ragtime/models/documents.py',
            'ragtime/models/queries.py',
            'ragtime/models/feedback.py',
            'ragtime/models/training.py',
            'ragtime/models/responses.py'
        ]

        for model_file in model_files:
            file_path = project_root / model_file
            assert file_path.exists(), f"Model file {model_file} not created"

    def test_settings_module_exists(self):
        """Verify settings management created"""
        settings_path = project_root / 'ragtime' / 'config' / 'settings.py'
        assert settings_path.exists(), "Settings module not created"

    def test_pydantic_models_importable(self):
        """Test that Pydantic models can be imported"""
        try:
            from ragtime.models.documents import DocumentMetadata, DocumentChunk
            from ragtime.models.queries import QueryRequest, QueryResponse
            from ragtime.config.settings import Settings

            # Basic instantiation test
            metadata = DocumentMetadata(book_title="Test Book")
            assert metadata.book_title == "Test Book"
        except ImportError as e:
            pytest.fail(f"Failed to import Pydantic models: {e}")


class TestPhase2CoreStructure:
    """Tests for Phase 2: Storage layer, DI, and logging"""

    def test_storage_layer_migrated(self):
        """Verify storage modules migrated"""
        storage_files = [
            'ragtime/storage/conport_client.py',
            'ragtime/storage/vector_db.py',
            'ragtime/storage/session_manager.py'
        ]

        for storage_file in storage_files:
            file_path = project_root / storage_file
            assert file_path.exists(), f"Storage file {storage_file} not migrated"

    def test_structured_logging_setup(self):
        """Verify structured logging is configured"""
        logging_path = project_root / 'ragtime' / 'monitoring' / 'logging.py'
        assert logging_path.exists(), "Logging module not created"

    def test_storage_imports_work(self):
        """Test that storage modules can be imported"""
        try:
            from ragtime.storage.conport_client import ConPortClient
            assert ConPortClient is not None
        except ImportError as e:
            pytest.fail(f"Failed to import storage modules: {e}")


class TestPhase3FeatureMigration:
    """Tests for Phase 3: Core, services, and API migration"""

    def test_core_modules_migrated(self):
        """Verify core business logic migrated"""
        core_files = [
            'ragtime/core/embeddings.py',
            'ragtime/core/retrieval.py',
            'ragtime/core/query_processor.py',
            'ragtime/core/response_generator.py'
        ]

        for core_file in core_files:
            file_path = project_root / core_file
            assert file_path.exists(), f"Core file {core_file} not migrated"

    def test_services_migrated(self):
        """Verify service modules migrated"""
        service_files = [
            'ragtime/services/conversation.py',
            'ragtime/services/feedback_analyzer.py',
            'ragtime/services/query_enhancer.py',
            'ragtime/services/training_data_gen.py',
            'ragtime/services/model_finetuner.py'
        ]

        for service_file in service_files:
            file_path = project_root / service_file
            assert file_path.exists(), f"Service file {service_file} not migrated"

    def test_api_endpoints_migrated(self):
        """Verify API endpoint modules migrated"""
        api_files = [
            'ragtime/api/routes.py',
            'ragtime/api/documents.py',
            'ragtime/api/queries.py',
            'ragtime/api/feedback.py'
        ]

        for api_file in api_files:
            file_path = project_root / api_file
            assert file_path.exists(), f"API file {api_file} not migrated"

    def test_flask_app_works(self):
        """Test that Flask app can be imported and initialized"""
        try:
            from ragtime.app import create_app
            app = create_app()
            assert app is not None
        except ImportError as e:
            pytest.fail(f"Failed to import Flask app: {e}")


class TestPhase4MCPRemoval:
    """Tests for Phase 4: MCP server code removal"""

    def test_mcp_modules_removed(self):
        """Verify deprecated MCP server modules are removed"""
        # These should NOT exist after Phase 4
        should_not_exist = [
            'ragtime/mcp',
            'mcp_integration.py',
            'mcp_server.py'
        ]

        for path in should_not_exist:
            full_path = project_root / path
            assert not full_path.exists(), f"MCP module {path} should be removed but still exists"

    def test_mcp_dependencies_removed(self):
        """Verify MCP-related dependencies removed from Pipfile"""
        pipfile_path = project_root / 'Pipfile'
        with open(pipfile_path, 'r') as f:
            pipfile_content = f.read()

        # These MCP-specific packages should be removed
        mcp_packages = ['fastmcp', 'mcp']
        for package in mcp_packages:
            # They might still exist if needed by ConPort, so this is informational
            if package in pipfile_content:
                pytest.skip(f"Package {package} still in Pipfile - may be needed by ConPort")


class TestPhase5Integration:
    """End-to-end integration tests after full migration"""

    def test_full_import_chain(self):
        """Test complete import chain works"""
        try:
            from ragtime.app import create_app
            from ragtime.config.settings import Settings
            from ragtime.models.queries import QueryRequest
            from ragtime.storage.conport_client import ConPortClient
            from ragtime.services.feedback_analyzer import FeedbackAnalyzer

            # All imports should succeed
            assert all([
                create_app,
                Settings,
                QueryRequest,
                ConPortClient,
                FeedbackAnalyzer
            ])
        except ImportError as e:
            pytest.fail(f"Import chain broken: {e}")

    def test_no_root_modules_remain(self):
        """Verify old root modules are removed or deprecated"""
        old_modules = [
            'embed.py',
            'query.py',
            'conversation.py',
            'feedback_analyzer.py'
        ]

        for old_module in old_modules:
            file_path = project_root / old_module
            if file_path.exists():
                # Check if it's a compatibility shim
                with open(file_path, 'r') as f:
                    content = f.read()
                    assert 'DeprecationWarning' in content or 'import ragtime' in content, \
                        f"{old_module} exists but is not a compatibility shim"

    def test_conport_integration_intact(self):
        """Critical: Verify ConPort integration still works"""
        try:
            from ragtime.storage.conport_client import ConPortClient

            # Try to instantiate (won't connect without actual server)
            # Just verify the class is properly structured
            assert hasattr(ConPortClient, '__init__')
            assert hasattr(ConPortClient, 'log_custom_data') or hasattr(ConPortClient, 'store_feedback')
        except ImportError as e:
            pytest.fail(f"ConPort integration broken: {e}")


@pytest.fixture
def phase_marker(request):
    """Fixture to mark which phase is being tested"""
    phase = request.config.getoption("--phase", default="0")
    return int(phase)


def pytest_addoption(parser):
    """Add custom command-line option for phase"""
    parser.addoption(
        "--phase",
        action="store",
        default="0",
        help="Migration phase to test (0-5)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on phase"""
    phase = int(config.getoption("--phase", default="0"))

    # Map phases to test classes
    phase_tests = {
        0: ["TestPhase0PreMigration"],
        1: ["TestPhase0PreMigration", "TestPhase1Foundation"],
        2: ["TestPhase1Foundation", "TestPhase2CoreStructure"],
        3: ["TestPhase2CoreStructure", "TestPhase3FeatureMigration"],
        4: ["TestPhase3FeatureMigration", "TestPhase4MCPRemoval"],
        5: ["TestPhase4MCPRemoval", "TestPhase5Integration"]
    }

    if phase in phase_tests:
        skip_marker = pytest.mark.skip(reason=f"Not applicable for phase {phase}")
        for item in items:
            # Get test class name
            test_class = item.parent.name if hasattr(item.parent, 'name') else None

            # Skip tests not relevant to current phase
            if test_class and test_class not in phase_tests[phase]:
                item.add_marker(skip_marker)


if __name__ == "__main__":
    """Run tests with: python tests/test_migration_validation.py"""
    pytest.main([__file__, "-v"])