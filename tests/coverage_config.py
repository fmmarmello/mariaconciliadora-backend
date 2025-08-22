"""
Test coverage configuration and reporting utilities.

This module provides utilities for:
- Configuring coverage reporting
- Generating coverage reports
- Setting coverage thresholds
- Creating coverage badges
"""

import os
import subprocess
import json
from pathlib import Path


class CoverageConfig:
    """Configuration for test coverage reporting."""
    
    # Coverage thresholds
    MINIMUM_COVERAGE = 85  # Minimum overall coverage percentage
    CRITICAL_MODULES_COVERAGE = 95  # Coverage for critical business logic modules
    
    # Critical modules that require higher coverage
    CRITICAL_MODULES = [
        'src/services/reconciliation_service.py',
        'src/services/duplicate_detection_service.py',
        'src/services/ai_service.py',
        'src/utils/validators.py',
        'src/utils/error_handler.py'
    ]
    
    # Modules to exclude from coverage
    EXCLUDE_PATTERNS = [
        '*/tests/*',
        '*/migrations/*',
        '*/venv/*',
        '*/env/*',
        '*/__pycache__/*',
        '*/test_*.py',
        '*_test.py'
    ]
    
    @classmethod
    def get_coverage_config(cls):
        """Get coverage configuration dictionary."""
        return {
            'run': {
                'source': ['src'],
                'omit': cls.EXCLUDE_PATTERNS,
                'branch': True,
                'parallel': True
            },
            'report': {
                'show_missing': True,
                'skip_covered': False,
                'sort': 'Cover',
                'precision': 2,
                'exclude_lines': [
                    'pragma: no cover',
                    'def __repr__',
                    'if self.debug:',
                    'if settings.DEBUG',
                    'raise AssertionError',
                    'raise NotImplementedError',
                    'if 0:',
                    'if __name__ == .__main__.:'
                ]
            },
            'html': {
                'directory': 'tests/coverage_html',
                'title': 'Maria Conciliadora Test Coverage Report'
            },
            'xml': {
                'output': 'tests/coverage.xml'
            },
            'json': {
                'output': 'tests/coverage.json'
            }
        }
    
    @classmethod
    def create_coverage_config_file(cls):
        """Create .coveragerc configuration file."""
        config_content = """[run]
source = src
omit = 
    */tests/*
    */migrations/*
    */venv/*
    */env/*
    */__pycache__/*
    */test_*.py
    *_test.py
branch = True
parallel = True

[report]
show_missing = True
skip_covered = False
sort = Cover
precision = 2
exclude_lines =
    pragma: no cover
    def __repr__
    if self.debug:
    if settings.DEBUG
    raise AssertionError
    raise NotImplementedError
    if 0:
    if __name__ == .__main__.:

[html]
directory = tests/coverage_html
title = Maria Conciliadora Test Coverage Report

[xml]
output = tests/coverage.xml

[json]
output = tests/coverage.json
"""
        
        with open('.coveragerc', 'w') as f:
            f.write(config_content)
        
        print("Created .coveragerc configuration file")


class CoverageReporter:
    """Utilities for generating and analyzing coverage reports."""
    
    def __init__(self):
        self.config = CoverageConfig()
    
    def run_coverage_tests(self):
        """Run tests with coverage collection."""
        try:
            # Run tests with coverage
            cmd = [
                'python', '-m', 'pytest',
                '--cov=src',
                '--cov-report=html:tests/coverage_html',
                '--cov-report=xml:tests/coverage.xml',
                '--cov-report=json:tests/coverage.json',
                '--cov-report=term-missing',
                '--cov-branch',
                'tests/'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Coverage tests completed successfully")
                print(result.stdout)
            else:
                print("❌ Coverage tests failed")
                print(result.stderr)
            
            return result.returncode == 0
        
        except Exception as e:
            print(f"Error running coverage tests: {e}")
            return False
    
    def analyze_coverage_report(self):
        """Analyze coverage report and check thresholds."""
        try:
            # Read JSON coverage report
            with open('tests/coverage.json', 'r') as f:
                coverage_data = json.load(f)
            
            # Extract overall coverage
            overall_coverage = coverage_data['totals']['percent_covered']
            
            print(f"\n📊 Coverage Analysis:")
            print(f"Overall Coverage: {overall_coverage:.2f}%")
            
            # Check overall threshold
            if overall_coverage >= self.config.MINIMUM_COVERAGE:
                print(f"✅ Overall coverage meets minimum threshold ({self.config.MINIMUM_COVERAGE}%)")
            else:
                print(f"❌ Overall coverage below minimum threshold ({self.config.MINIMUM_COVERAGE}%)")
            
            # Check critical modules
            print(f"\n🔍 Critical Modules Coverage:")
            critical_issues = []
            
            for module in self.config.CRITICAL_MODULES:
                if module in coverage_data['files']:
                    module_coverage = coverage_data['files'][module]['summary']['percent_covered']
                    print(f"  {module}: {module_coverage:.2f}%")
                    
                    if module_coverage < self.config.CRITICAL_MODULES_COVERAGE:
                        critical_issues.append(f"{module}: {module_coverage:.2f}%")
                else:
                    print(f"  {module}: Not found in coverage report")
                    critical_issues.append(f"{module}: Not tested")
            
            if critical_issues:
                print(f"\n❌ Critical modules below threshold ({self.config.CRITICAL_MODULES_COVERAGE}%):")
                for issue in critical_issues:
                    print(f"  - {issue}")
            else:
                print(f"\n✅ All critical modules meet coverage threshold")
            
            # Generate summary
            self._generate_coverage_summary(coverage_data)
            
            return len(critical_issues) == 0 and overall_coverage >= self.config.MINIMUM_COVERAGE
        
        except FileNotFoundError:
            print("❌ Coverage report not found. Run coverage tests first.")
            return False
        except Exception as e:
            print(f"Error analyzing coverage report: {e}")
            return False
    
    def _generate_coverage_summary(self, coverage_data):
        """Generate coverage summary report."""
        summary = {
            'timestamp': coverage_data['meta']['timestamp'],
            'overall_coverage': coverage_data['totals']['percent_covered'],
            'lines_covered': coverage_data['totals']['covered_lines'],
            'lines_total': coverage_data['totals']['num_statements'],
            'branches_covered': coverage_data['totals']['covered_branches'],
            'branches_total': coverage_data['totals']['num_branches'],
            'files_analyzed': len(coverage_data['files']),
            'critical_modules': {}
        }
        
        # Add critical modules data
        for module in self.config.CRITICAL_MODULES:
            if module in coverage_data['files']:
                summary['critical_modules'][module] = {
                    'coverage': coverage_data['files'][module]['summary']['percent_covered'],
                    'lines_covered': coverage_data['files'][module]['summary']['covered_lines'],
                    'lines_total': coverage_data['files'][module]['summary']['num_statements']
                }
        
        # Save summary
        with open('tests/coverage_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Coverage Summary saved to tests/coverage_summary.json")
    
    def generate_coverage_badge(self):
        """Generate coverage badge for README."""
        try:
            with open('tests/coverage.json', 'r') as f:
                coverage_data = json.load(f)
            
            coverage_percent = coverage_data['totals']['percent_covered']
            
            # Determine badge color based on coverage
            if coverage_percent >= 90:
                color = 'brightgreen'
            elif coverage_percent >= 80:
                color = 'green'
            elif coverage_percent >= 70:
                color = 'yellow'
            elif coverage_percent >= 60:
                color = 'orange'
            else:
                color = 'red'
            
            # Generate badge URL
            badge_url = f"https://img.shields.io/badge/coverage-{coverage_percent:.1f}%25-{color}"
            
            # Generate badge markdown
            badge_markdown = f"![Coverage]({badge_url})"
            
            # Save badge info
            badge_info = {
                'coverage_percent': coverage_percent,
                'color': color,
                'url': badge_url,
                'markdown': badge_markdown
            }
            
            with open('tests/coverage_badge.json', 'w') as f:
                json.dump(badge_info, f, indent=2)
            
            print(f"\n🏷️  Coverage Badge Generated:")
            print(f"Coverage: {coverage_percent:.1f}%")
            print(f"Badge URL: {badge_url}")
            print(f"Markdown: {badge_markdown}")
            
            return badge_info
        
        except Exception as e:
            print(f"Error generating coverage badge: {e}")
            return None
    
    def create_coverage_report_html(self):
        """Create enhanced HTML coverage report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Maria Conciliadora - Test Coverage Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { display: flex; gap: 20px; margin: 20px 0; }
        .metric { background: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }
        .metric h3 { margin: 0; color: #2c3e50; }
        .metric .value { font-size: 24px; font-weight: bold; color: #3498db; }
        .critical { background: #fff3cd; border: 1px solid #ffeaa7; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; }
        .warning { background: #f8d7da; border: 1px solid #f5c6cb; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #f8f9fa; }
        .high-coverage { color: #28a745; }
        .medium-coverage { color: #ffc107; }
        .low-coverage { color: #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Maria Conciliadora - Test Coverage Report</h1>
        <p>Generated on: {timestamp}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Overall Coverage</h3>
            <div class="value">{overall_coverage:.1f}%</div>
        </div>
        <div class="metric">
            <h3>Lines Covered</h3>
            <div class="value">{lines_covered}/{lines_total}</div>
        </div>
        <div class="metric">
            <h3>Branches Covered</h3>
            <div class="value">{branches_covered}/{branches_total}</div>
        </div>
        <div class="metric">
            <h3>Files Analyzed</h3>
            <div class="value">{files_analyzed}</div>
        </div>
    </div>
    
    <h2>Critical Modules Coverage</h2>
    <table>
        <thead>
            <tr>
                <th>Module</th>
                <th>Coverage</th>
                <th>Lines Covered</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {critical_modules_rows}
        </tbody>
    </table>
    
    <p><a href="coverage_html/index.html">View Detailed Coverage Report</a></p>
</body>
</html>
        """
        
        try:
            with open('tests/coverage_summary.json', 'r') as f:
                summary = json.load(f)
            
            # Generate critical modules rows
            critical_rows = []
            for module, data in summary['critical_modules'].items():
                coverage = data['coverage']
                status_class = 'high-coverage' if coverage >= 95 else 'medium-coverage' if coverage >= 85 else 'low-coverage'
                status_text = '✅ Excellent' if coverage >= 95 else '⚠️ Good' if coverage >= 85 else '❌ Needs Improvement'
                
                row = f"""
                <tr>
                    <td>{module}</td>
                    <td class="{status_class}">{coverage:.1f}%</td>
                    <td>{data['lines_covered']}/{data['lines_total']}</td>
                    <td>{status_text}</td>
                </tr>
                """
                critical_rows.append(row)
            
            # Generate HTML
            html_content = html_template.format(
                timestamp=summary['timestamp'],
                overall_coverage=summary['overall_coverage'],
                lines_covered=summary['lines_covered'],
                lines_total=summary['lines_total'],
                branches_covered=summary['branches_covered'],
                branches_total=summary['branches_total'],
                files_analyzed=summary['files_analyzed'],
                critical_modules_rows=''.join(critical_rows)
            )
            
            # Save HTML report
            os.makedirs('tests/reports', exist_ok=True)
            with open('tests/reports/coverage_report.html', 'w') as f:
                f.write(html_content)
            
            print("📄 Enhanced HTML coverage report created: tests/reports/coverage_report.html")
        
        except Exception as e:
            print(f"Error creating HTML coverage report: {e}")


def main():
    """Main function to run coverage analysis."""
    print("🧪 Maria Conciliadora - Test Coverage Analysis")
    print("=" * 50)
    
    # Create coverage configuration
    CoverageConfig.create_coverage_config_file()
    
    # Initialize reporter
    reporter = CoverageReporter()
    
    # Run coverage tests
    print("\n1. Running tests with coverage...")
    if not reporter.run_coverage_tests():
        print("❌ Coverage tests failed. Exiting.")
        return False
    
    # Analyze coverage
    print("\n2. Analyzing coverage report...")
    coverage_ok = reporter.analyze_coverage_report()
    
    # Generate badge
    print("\n3. Generating coverage badge...")
    reporter.generate_coverage_badge()
    
    # Create enhanced HTML report
    print("\n4. Creating enhanced HTML report...")
    reporter.create_coverage_report_html()
    
    print("\n" + "=" * 50)
    if coverage_ok:
        print("✅ All coverage requirements met!")
    else:
        print("❌ Coverage requirements not met. Please improve test coverage.")
    
    return coverage_ok


if __name__ == '__main__':
    main()