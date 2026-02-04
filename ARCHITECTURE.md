# 🏗️ System Architecture

## Overview

The Space Domain Awareness AI System is designed as a modular, scalable pipeline for tracking and predicting space objects with defense-grade reliability.

## Design Principles

1. **Modularity**: Each phase is independent and testable
2. **Uncertainty-Aware**: All outputs include confidence bounds
3. **Explainability**: ML decisions are interpretable
4. **Validation-First**: Test scenarios drive implementation
5. **Operational Focus**: Designed for operator decision-making

## System Layers

### Layer 1: Simulation (Data Generation)
**Purpose**: Generate realistic synthetic sensor data

**Components**:
- Orbital propagator (SGP4/SDP4)
- Sensor models (radar, optical)
- Noise injection
- Ground truth logging

**Inputs**: TLE files, sensor configurations  
**Outputs**: Time-series measurements, ground truth trajectories

**Key Files**:
- `src/simulation/orbital_mechanics.py` - SGP4 propagation
- `src/simulation/sensor_models.py` - Sensor simulation
- `src/simulation/data_generator.py` - Dataset creation
- `src/simulation/noise_models.py` - Measurement uncertainty

---

### Layer 2: Tracking (State Estimation)
**Purpose**: Maintain tracks of space objects

**Components**:
- Kalman filters (EKF/UKF)
- Data association (GNN, Hungarian)
- Track management
- Maneuver detection

**Inputs**: Sensor measurements  
**Outputs**: Track states with covariances

**Key Files**:
- `src/tracking/kalman_filters.py` - EKF/UKF implementation
- `src/tracking/data_association.py` - Track-to-measurement matching
- `src/tracking/track_manager.py` - Track lifecycle
- `src/tracking/maneuver_detection.py` - Anomaly detection

---

### Layer 3: ML Prediction (Forecasting & Classification)
**Purpose**: Predict future states and classify objects

**Components**:
- Trajectory predictor (Transformer)
- Object classifier (Ensemble)
- Threat scorer
- Uncertainty quantification

**Inputs**: Track histories  
**Outputs**: Predictions with uncertainty, classifications, threat scores

**Key Files**:
- `src/ml/models/trajectory_predictor.py` - Transformer model
- `src/ml/models/object_classifier.py` - Classification ensemble
- `src/ml/features/engineering.py` - Feature extraction
- `src/ml/training/train_predictor.py` - Training pipeline

---

### Layer 4: Operational (User Interface)
**Purpose**: Present information to operators

**Components**:
- FastAPI backend
- React + CesiumJS frontend
- Alert system
- Query interface

**Inputs**: Tracks, predictions, alerts  
**Outputs**: Visualizations, operator decisions

**Key Files**:
- `src/api/main.py` - FastAPI application
- `src/api/routes/tracks.py` - Track queries
- `src/dashboard/src/components/CesiumViewer.tsx` - 3D visualization

---

### Layer 5: Validation (Testing & V&V)
**Purpose**: Ensure system reliability

**Components**:
- Unit tests
- Integration tests
- Scenario tests
- Performance benchmarks

**Inputs**: All system components  
**Outputs**: Test reports, validation metrics

**Key Files**:
- `tests/unit/` - Component tests
- `tests/integration/` - End-to-end tests
- `tests/scenarios/` - Operational scenarios
- `tests/benchmarks/` - Performance tests

## Data Flow

```
TLE Data → Propagation → Sensor Simulation → Measurements
                                                  ↓
                                            Tracking Engine
                                                  ↓
                                          Track States + History
                                                  ↓
                                            ML Prediction
                                                  ↓
                                    Predictions + Classifications
                                                  ↓
                                          Operational Dashboard
                                                  ↓
                                          Operator Decisions
```

## Technology Stack

### Core Libraries
- **Python 3.10+**: Primary language
- **NumPy/SciPy**: Numerical computing
- **Pandas/Polars**: Data manipulation

### Domain-Specific
- **Skyfield**: Orbital mechanics
- **SGP4**: Satellite propagation
- **FilterPy**: Kalman filtering

### Machine Learning
- **PyTorch**: Deep learning framework
- **XGBoost**: Gradient boosting
- **scikit-learn**: Classical ML
- **SHAP**: Model explainability

### Backend/API
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **SQLAlchemy**: Database ORM
- **Redis**: Caching layer

### Frontend
- **React**: UI framework
- **TypeScript**: Type-safe JavaScript
- **CesiumJS**: 3D geospatial visualization

### Testing
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **pytest-benchmark**: Performance testing

## Deployment Architecture

(To be defined in Phase 4)

### Containerization
- Docker containers for each service
- Docker Compose for local development
- Kubernetes for production scaling

### Services
- **API Service**: FastAPI backend
- **ML Service**: Model inference
- **Database**: PostgreSQL + TimescaleDB
- **Cache**: Redis
- **Frontend**: React SPA

## Security Considerations

- No classified data (all synthetic)
- API authentication for production
- Rate limiting on endpoints
- Input validation throughout
- Secure configuration management

## Scalability Considerations

- Horizontal scaling via containerization
- Database sharding for large object catalogs
- Caching for frequently accessed tracks
- Async processing for ML inference
- Load balancing for API endpoints

## Performance Requirements

- **Tracking Latency**: <100ms per update cycle
- **API Response Time**: <100ms for queries
- **ML Inference**: <50ms per prediction
- **Database Queries**: <10ms for indexed lookups
- **Real-time Updates**: 1Hz minimum

## Monitoring & Observability

- Structured JSON logging
- Prometheus metrics
- Performance dashboards
- Error tracking
- Audit trails for operator actions

---

**Last Updated**: 2026-02-03  
**Version**: 0.1.0
