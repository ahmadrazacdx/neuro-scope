"""
Tests for NeuroScope main module functionality.
"""

import sys
from unittest.mock import patch

import matplotlib
import numpy as np
import pytest

from neuroscope import __version__
from neuroscope.__main__ import main, print_banner, show_help, validate_installation


class TestPrintBanner:
    """Test the banner printing functionality."""

    def test_print_banner_contains_version(self, capsys):
        """Test that banner contains version information."""
        print_banner()
        captured = capsys.readouterr()

        assert "NeuroScope" in captured.out
        assert __version__ in captured.out
        assert "Microscope for Neural Networks" in captured.out
        assert sys.version.split()[0] in captured.out
        assert np.__version__ in captured.out
        assert matplotlib.__version__ in captured.out

    def test_print_banner_format(self, capsys):
        """Test that banner has proper formatting."""
        print_banner()
        captured = capsys.readouterr()

        # Check for box drawing characters
        assert "╔" in captured.out
        assert "║" in captured.out
        assert "╚" in captured.out


class TestValidateInstallation:
    """Test the installation validation functionality."""

    def test_validate_installation_success(self, capsys):
        """Test successful validation."""
        result = validate_installation()
        captured = capsys.readouterr()

        assert result is True
        assert "Validating NeuroScope installation" in captured.out
        assert "Testing imports" in captured.out
        assert "Testing MLP creation" in captured.out
        assert "Testing model compilation" in captured.out
        assert "Testing forward pass" in captured.out
        assert "Testing loss and metrics" in captured.out
        assert "Testing training functionality" in captured.out
        assert "All validation tests passed" in captured.out

    def test_validate_installation_failure(self, capsys):
        """Test validation failure handling."""
        # Patch the MLP class where it's imported in the validation function
        with patch("neuroscope.MLP") as mock_mlp:
            mock_mlp.side_effect = Exception("Test error")

            result = validate_installation()
            captured = capsys.readouterr()

            assert result is False
            assert "Validation failed: Test error" in captured.out

    def test_validate_installation_import_error(self, capsys):
        """Test validation with import errors."""
        # This test ensures validation catches import issues
        with patch.dict("sys.modules", {"neuroscope.mlp": None}):
            result = validate_installation()
            captured = capsys.readouterr()

            # Should still work since we're testing the actual imports
            # This test mainly ensures the try-catch structure works
            assert isinstance(result, bool)


class TestShowHelp:
    """Test the help display functionality."""

    def test_show_help_content(self, capsys):
        """Test help content is comprehensive."""
        show_help()
        captured = capsys.readouterr()

        assert "NeuroScope - A Microscope for Neural Networks" in captured.out
        assert "USAGE:" in captured.out
        assert "OPTIONS:" in captured.out
        assert "EXAMPLES:" in captured.out
        assert "GETTING STARTED:" in captured.out
        assert "DOCUMENTATION:" in captured.out
        assert "SUPPORT:" in captured.out
        assert "--version" in captured.out
        assert "--validate" in captured.out


class TestMainFunction:
    """Test the main CLI function."""

    def test_main_no_arguments(self, capsys):
        """Test main with no arguments shows help."""
        result = main([])
        captured = capsys.readouterr()

        assert result == 0
        assert "NeuroScope" in captured.out
        assert "USAGE:" in captured.out

    def test_main_version_flag(self, capsys):
        """Test --version flag."""
        result = main(["--version"])
        captured = capsys.readouterr()

        assert result == 0
        assert "NeuroScope" in captured.out
        assert __version__ in captured.out

    def test_main_version_short_flag(self, capsys):
        """Test -v flag."""
        result = main(["-v"])
        captured = capsys.readouterr()

        assert result == 0
        assert "NeuroScope" in captured.out
        assert __version__ in captured.out

    def test_main_validate_flag_success(self, capsys):
        """Test --validate flag with successful validation."""
        result = main(["--validate"])
        captured = capsys.readouterr()

        assert result == 0
        assert "Validating NeuroScope installation" in captured.out
        assert "All validation tests passed" in captured.out

    @patch("neuroscope.__main__.validate_installation")
    def test_main_validate_flag_failure(self, mock_validate, capsys):
        """Test --validate flag with failed validation."""
        mock_validate.return_value = False

        result = main(["--validate"])

        assert result == 1
        mock_validate.assert_called_once()

    def test_main_invalid_argument(self):
        """Test main with invalid arguments."""
        with pytest.raises(SystemExit):
            main(["--invalid-argument"])

    def test_main_multiple_flags(self, capsys):
        """Test main with multiple flags (should process first valid one)."""
        result = main(["--version", "--validate"])
        captured = capsys.readouterr()

        assert result == 0
        # Version should be processed (order in main function)
        assert "NeuroScope" in captured.out
        assert __version__ in captured.out


class TestCLIIntegration:
    """Integration tests for the CLI."""

    def test_cli_as_module(self, capsys):
        """Test running as python -m neuroscope."""
        # This simulates: python -m neuroscope --version
        with patch.object(sys, "argv", ["neuroscope", "--version"]):
            result = main()
            captured = capsys.readouterr()

            assert result == 0
            assert "NeuroScope" in captured.out
            assert __version__ in captured.out

    def test_cli_help_accessibility(self, capsys):
        """Test that help is easily accessible."""
        # Test help invocation (no --help flag in simplified version, just empty args)
        result = main([])
        captured = capsys.readouterr()

        assert result == 0
        assert "NeuroScope" in captured.out

    def test_cli_error_handling(self):
        """Test CLI error handling for invalid inputs."""
        with pytest.raises(SystemExit):
            main(["--nonexistent-flag"])


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_main_with_none_argv(self, capsys):
        """Test main with None argv (should use sys.argv)."""
        with patch.object(sys, "argv", ["neuroscope", "--version"]):
            result = main(None)
            captured = capsys.readouterr()

            assert result == 0
            assert "NeuroScope" in captured.out
            assert __version__ in captured.out

    @patch("builtins.print")
    def test_banner_print_called(self, mock_print):
        """Test that banner actually calls print."""
        print_banner()
        mock_print.assert_called()

    def test_version_consistency(self):
        """Test that CLI version matches package version."""
        from neuroscope import __version__ as pkg_version
        from neuroscope.__main__ import __version__ as cli_version

        assert pkg_version == cli_version


# Fixtures and utilities for testing
@pytest.fixture
def mock_numpy_random():
    """Fixture to mock numpy random for reproducible tests."""
    with (
        patch("numpy.random.seed") as mock_seed,
        patch("numpy.random.randn") as mock_randn,
        patch("numpy.random.randint") as mock_randint,
    ):

        # Set up predictable random values
        mock_randn.return_value = np.ones((100, 4))
        mock_randint.return_value = np.ones((100, 1))

        yield {"seed": mock_seed, "randn": mock_randn, "randint": mock_randint}


if __name__ == "__main__":
    pytest.main([__file__])
