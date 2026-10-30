# HealthFlux - Smart Healthcare Analytics Platform

A comprehensive healthcare analytics platform that combines traditional SQL-based analysis with PySpark-powered big data processing to deliver insights from healthcare datasets.

## Overview

HealthFlux is a full-stack healthcare analytics solution that processes and visualizes medical data through multiple analytical approaches. The platform demonstrates the integration of traditional relational databases with modern big data processing frameworks.

## Architecture

```
HealthFlux/
├── healthcare-backend/     # Node.js/Express API server
├── healthcare-frontend/    # React.js frontend with Chart.js
├── db_populate.py          # PostgreSQL data population script
├── pyspark_analysis.py     # PySpark analytics engine
├── spark_api.py            # Flask API for Spark results
└── schema.sql              # Database schema definition
```

## Functionalities Implemented

### 1. Data Management & Storage

* **PostgreSQL Database**: Structured healthcare data storage with optimized schema
* **Data Population**: Automated CSV ingestion with data cleaning and normalization
* **Relational Modeling**: Patients, Doctors, Hospitals, Appointments, MedicalRecords

### 2. Dual Analytics Engine

#### SQL-Based Analytics (Traditional Approach)

* Top Diagnoses: Identify most common medical conditions
* Patient Statistics: Age distribution, gender analysis
* Hospital Utilization: Appointment patterns and capacity analysis
* Readmission Rates: Track patients with multiple visits

#### PySpark Analytics (Big Data Approach)

* Distributed Processing: Handle large-scale healthcare datasets
* Advanced Analytics: Complex aggregations and pattern recognition
* JSON Output: Structured results for frontend consumption
* Scalable Architecture: Can process terabytes of healthcare data

### 3. API Layer

* **RESTful APIs**: Multiple endpoints for different data sources
* **Dual Backend Support**:

  * Node.js/Express for SQL queries
  * Flask for PySpark results
* **CORS Enabled**: Cross-origin resource sharing for frontend integration

### 4. Frontend Visualization

* Interactive Charts: Diagnosis distributions, hospital usage, readmission stats
* Real-time Data: Live updates from analytics engines
* Responsive Design: Tailwind CSS for modern UI
* Chart.js Integration: Professional data visualization

## Technical Stack

### Backend

* Node.js + Express (Primary API server)
* PostgreSQL (Relational database)
* Python + Flask (Spark results API)
* PySpark (Big data processing)
* psycopg2 (PostgreSQL Python adapter)

### Frontend

* React.js (Component-based UI)
* Vite (Build tool and dev server)
* Chart.js (Data visualization)
* Tailwind CSS (Utility-first styling)

### Analytics

* SQL: Traditional relational queries
* PySpark: Distributed data processing
* Pandas: Data manipulation for ETL

## Extended PySpark Use Cases

The current implementation demonstrates basic PySpark analytics, but the architecture supports extensive healthcare analytics.

### Current PySpark Features

* Top 5 medical condition analysis
* Readmission rate calculations
* Patient admission pattern recognition

### Extended PySpark Capabilities

#### 1. Predictive Analytics

```python
# Patient readmission prediction
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler  

features = ['age', 'admission_count', 'medical_condition_severity']
assembler = VectorAssembler(inputCols=features, outputCol='features')
```

#### 2. Temporal Analysis

```python
# Seasonal disease patterns
from pyspark.sql.functions import month, year  

seasonal_analysis = df.groupBy(
    year('date_of_admission').alias('year'),
    month('date_of_admission').alias('month'),
    'medical_condition'
).count()
```

#### 3. Real-time Stream Processing

```python
# Live patient monitoring
streaming_df = spark.readStream.schema(healthcare_schema).json("kafka://...")
windowed_counts = streaming_df.groupBy(
    window("timestamp", "5 minutes"),
    "medical_condition"
).count()
```

## Setup Instructions

### Prerequisites

* PostgreSQL
* Python 3.8+
* Node.js 16+
* Java 11 (for PySpark)
* Apache Spark

### Installation

1. **Database Setup**

```bash
psql -U postgres -c "CREATE DATABASE healthcare_db;"
psql -U postgres -d healthcare_db -f schema.sql
python db_populate.py
```

2. **Backend Services**

```bash
# Node.js Backend
cd healthcare-backend
npm install
npm start  

# PySpark API
python spark_api.py  

# Run PySpark Analysis
python pyspark_analysis.py
```

3. **Frontend**

```bash
cd healthcare-frontend
npm install
npm run dev
```

## API Endpoints

### SQL Analytics

* `GET /api/patients` – Patient records
* `GET /api/analytics/diagnoses` – Top diagnoses (SQL)
* `GET /api/analytics/sql/top-diagnoses` – Alternative SQL endpoint

### Spark Analytics

* `GET /api/analytics/spark/top-diagnoses` – PySpark processed results
* `GET /api/spark/readmissions` – Readmission analytics

## Future Enhancements

### Advanced PySpark Implementations

1. **Machine Learning Pipeline**

   * Disease outbreak prediction
   * Patient risk stratification
   * Treatment outcome forecasting
2. **Real-time Analytics**

   * Live patient monitoring dashboard
   * Anomaly detection in vital signs
   * Emergency room optimization
3. **Big Data Scale**

   * Hadoop integration for petabyte-scale data
   * Cloud deployment (AWS EMR, Databricks)
   * Streaming data from IoT medical devices
4. **Advanced Visualizations**

   * Real-time dashboards
   * Geographic heat maps
   * Predictive trend lines

## Performance Comparison

| Aspect            | SQL Approach         | PySpark Approach         |
| ----------------- | -------------------- | ------------------------ |
| Data Scale        | GBs                  | TBs+                     |
| Processing Speed  | Fast for medium data | Optimized for large data |
| Complex Analytics | Limited              | Advanced ML capabilities |
| Real-time         | Batch processing     | Streaming possible       |
| Cost              | Lower infrastructure | Higher (cluster needed)  |

## Use Cases

* **Hospital Administrators**: Capacity planning and resource allocation
* **Medical Researchers**: Disease pattern analysis and clinical studies
* **Healthcare Providers**: Patient readmission reduction strategies
* **Insurance Companies**: Risk assessment and fraud detection
* **Public Health Officials**: Epidemic monitoring and prevention

## Contributing

This project demonstrates modern healthcare analytics architecture and can be extended for:

* Additional data sources (EHR systems, wearables)
* More complex analysis of data (over various intervals)
* Real-time streaming analytics
* Multi-hospital federation analytics
* FP Growth to understand correlations among diseases

---

**HealthFlux** bridges the gap between traditional healthcare databases and modern big data analytics, providing a scalable foundation for the next generation of healthcare intelligence systems.
