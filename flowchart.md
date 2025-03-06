# Stock Model System Flow

```mermaid
flowchart TB
    subgraph User Interface
        A[Web Interface] --> B[Flask App: run_analysis()]
        B --> C{Market Selection}
    end

    subgraph Data Collection
        C -->|UK Market| D[UK Stock Collector]
        C -->|US Market| E[US Stock Collector]
        D --> |collect_historical_data()| F[Historical Data]
        E --> |collect_historical_data()| F
        D --> |_get_news_sentiment()| G[News Sentiment]
        E --> |_get_news_sentiment()| G
    end

    subgraph Processing
        F --> H[Feature Engineering: generate_features()]
        G --> H
        H --> I[Model Training: train_model()]
        I --> J[Stock Predictor]
    end

    subgraph Output
        J --> |predict_top_gainers()| K[Top Gainers Prediction]
        K --> |generate_analysis_report()| L[Analysis Report]
        L --> M[Web Display]
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style C fill:#fda,stroke:#333,stroke-width:2px
    style D fill:#dfd,stroke:#333,stroke-width:2px
    style E fill:#dfd,stroke:#333,stroke-width:2px
    style F fill:#ddf,stroke:#333,stroke-width:2px
    style G fill:#ddf,stroke:#333,stroke-width:2px
    style H fill:#fdd,stroke:#333,stroke-width:2px
    style I fill:#fdd,stroke:#333,stroke-width:2px
    style J fill:#fdd,stroke:#333,stroke-width:2px
    style K fill:#dff,stroke:#333,stroke-width:2px
    style L fill:#dff,stroke:#333,stroke-width:2px
    style M fill:#dff,stroke:#333,stroke-width:2px
```

This flowchart visualizes the stock model system's architecture and data flow:

1. User Interface Layer:
   - Web interface for user interaction
   - Flask application handling requests
   - Market selection (UK/US)

2. Data Collection Layer:
   - Market-specific stock collectors
   - Historical data gathering
   - News sentiment analysis

3. Processing Layer:
   - Feature engineering
   - Model training
   - Stock prediction

4. Output Layer:
   - Top gainers prediction
   - Analysis report generation
   - Web display of results