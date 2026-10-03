// Jenkins pipeline for Torch_Sim_Frontend.
//
// Stages: clean old reports -> setup (CPU-only PyTorch, reused across builds) -> lint (ruff) -> tests (pytest,
// JUnit) with coverage (Cobertura) -> the arithmetic, drift and speed gate against
// ci/perf_baseline.json -> the requirements traceability matrix -> results.md -> an optional nightly sweep over models and lengths.
//
// Needs on the agent: Python 3.10+ and network access to PyPI and download.pytorch.org.
// Plugins: Pipeline, Git, JUnit, Coverage.

pipeline {
    agent any

    parameters {
        booleanParam(name: 'NIGHTLY', defaultValue: false, description: 'Also run the model x length sweep')
        string(name: 'PERF_MARGIN', defaultValue: '0.5',
               description: 'Allowed slow-down of trace capture against ci/perf_baseline.json')
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '30'))
        timeout(time: 60, unit: 'MINUTES')
    }

    stages {
        // The workspace is reused between builds (it keeps the virtualenv and build caches), so
        // delete the previous build's reports first. Without this a build that fails before its
        // tests run publishes the last build's JUnit results as its own (Rust_DES_Kernel #4 did).
        stage('Clean reports') {
            steps {
                sh 'rm -f pytest-junit.xml coverage.xml perf_report.md sweep.csv'
            }
        }

        stage('Setup') {
            steps {
                // The venv (with a ~700 MB CPU PyTorch) is kept in the workspace between builds.
                sh '''
                    [ -x .venv/bin/python ] || python3 -m venv .venv
                    .venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
                    .venv/bin/pip install -q -e ".[test]" ruff
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '.venv/bin/ruff check src tests ci examples'
            }
        }

        stage('Tests') {
            steps {
                sh '.venv/bin/pytest -p no:logging --junitxml=pytest-junit.xml --cov=simfront --cov-report=xml:coverage.xml'
                recordCoverage(tools: [[parser: 'COBERTURA', pattern: 'coverage.xml']], sourceCodeRetention: 'LAST_BUILD')
            }
        }

        stage('Arithmetic, drift and speed gate') {
            steps {
                sh ".venv/bin/python ci/perf_gate.py --margin ${params.PERF_MARGIN ?: '0.5'}"
            }
            post {
                always { archiveArtifacts artifacts: 'perf_report.md', allowEmptyArchive: true }
            }
        }

        stage('Traceability') {
            steps {
                sh '.venv/bin/python ci/trace_matrix.py > /dev/null'
                archiveArtifacts artifacts: 'docs/traceability.md'
            }
        }

        stage('Results') {
            steps {
                sh '.venv/bin/python examples/results.py > /dev/null'
                archiveArtifacts artifacts: 'examples/results.md'
            }
        }

        stage('Nightly sweep') {
            when { expression { params.NIGHTLY } }
            steps {
                sh '.venv/bin/python ci/sweep.py'
                archiveArtifacts artifacts: 'sweep.csv'
            }
        }
    }

    post {
        always {
            junit testResults: 'pytest-junit.xml', allowEmptyResults: true
        }
    }
}
