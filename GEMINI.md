# GEMINI.md

## Project Overview

This project, **Maria Conciliadora**, is a full-stack financial dashboard application. It's designed to consolidate banking information, provide interactive data visualizations, and leverage AI for insights and reconciliation.

The application consists of two main parts:

*   **Backend:** A Python-based backend built with the **Flask** web framework. It handles the business logic, including:
    *   Processing OFX and XLSX files for financial data.
    *   Storing data in a SQLite database.
    *   Providing a RESTful API for the frontend.
    *   Integrating with AI services (OpenAI, Groq) for data analysis.

*   **Frontend:** A modern user interface built with **React** and **Vite**. It features:
    *   A clean, interactive dashboard for data visualization.
    *   Components from the Shadcn/UI library.
    *   Styling with Tailwind CSS.
    *   Charts and graphs using Recharts.

The project is well-structured, with separate directories for the backend and frontend, and clear documentation in the `README.md` file.

## Functionalities

### Backend

*   **OFX and XLSX Processing:**
    *   Parses and validates OFX and XLSX files to extract financial transactions.
    *   Identifies the bank from the OFX file content.
    *   Detects and handles duplicate transactions.

*   **AI Services:**
    *   Categorizes transactions using a rule-based system and a custom-trained model.
    *   Detects anomalies in transactions using the Isolation Forest algorithm.
    *   Generates financial insights and recommendations.
    *   Provides financial forecasting based on historical data.

*   **Reconciliation:**
    *   Matches bank transactions with company financial entries.
    *   Calculates a match score based on amount, date, and description similarity.
    *   Allows for manual confirmation or rejection of matches.

*   **API:**
    *   Provides a comprehensive set of RESTful endpoints for all frontend functionalities.
    *   Includes endpoints for file uploads, data retrieval, insights, and reconciliation.

### Frontend

*   **Dashboard:**
    *   Displays a summary of total income, expenses, and net balance.
    *   Visualizes spending by category and transactions by bank through interactive charts.
    *   Lists recent transactions.

*   **File Upload:**
    *   Allows users to upload OFX and XLSX files via drag-and-drop or file selection.
    *   Provides real-time feedback on the upload and processing status.

*   **Transactions List:**
    *   Displays a paginated list of all transactions.
    *   Offers advanced filtering options by description, bank, category, type, and date range.

*   **Insights Panel:**
    *   Presents statistical insights and patterns identified from the financial data.
    *   Generates and displays AI-powered insights and recommendations.

*   **Financial Tracker:**
    *   Allows for the upload and management of company financial data from XLSX files.
    *   Provides a view for correcting incomplete or incorrect entries.

*   **AI Training:**
    *   Enables users to train the AI model with their own financial data for improved categorization accuracy.
    *   Displays performance metrics of the trained model.

*   **Financial Predictions:**
    *   Shows financial forecasts for future months based on historical data.
    *   Visualizes predicted income, expenses, and net flow in a line chart.

*   **Reconciliation:**
    *   Presents pending reconciliation matches for user review.
    *   Allows users to confirm or reject matches.
    *   Displays a summary report of the reconciliation status.

## Building and Running

### Backend (Flask)

1.  **Navigate to the backend directory:**
    ```bash
    cd mariaconciliadora-backend
    ```

2.  **Activate the Python virtual environment:**
    ```bash
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Flask application:**
    ```bash
    python src/main.py
    ```

The backend will be running at `http://localhost:5000`.

### Frontend (React)

1.  **Navigate to the frontend directory:**
    ```bash
    cd mariaconciliadora-frontend
    ```

2.  **Install the required dependencies:**
    ```bash
    pnpm install
    ```

3.  **Run the frontend application in development mode:**
    ```bash
    pnpm run dev --host
    ```

The frontend will be running at `http://localhost:5173`.

### Integrated Application

To run the application as a single unit, with the frontend served by the Flask backend:

1.  **Build the frontend:**
    ```bash
    cd mariaconciliadora-frontend
    pnpm run build
    ```

2.  **Copy the frontend build to the backend's static folder:**
    ```bash
    cd ..
    rm -rf mariaconciliadora-backend/src/static/*
    cp -r mariaconciliadora-frontend/dist/* mariaconciliadora-backend/src/static/
    ```

3.  **Run the Flask application:**
    ```bash
    cd mariaconciliadora-backend
    source venv/bin/activate
    python src/main.py
    ```

The integrated application will be accessible at `http://localhost:5000`.

## Development Conventions

*   **Backend:**
    *   The backend follows a modular structure, with blueprints for different routes (`user`, `transactions`).
    *   It uses SQLAlchemy as the ORM for database interactions.
    *   Environment variables are used for configuration (e.g., database URL, API keys).
    *   The `main.py` file serves as the entry point for the application.

*   **Frontend:**
    *   The frontend uses `vite` for fast development and builds.
    *   It uses `pnpm` as the package manager.
    *   The project is configured with ESLint for code linting.
    *   The `vite.config.js` file includes an alias `@` for the `src` directory.

*   **AI Integration:**
    *   The application is designed to work with AI models from OpenAI (GPT-4o-mini) and Groq (llama3-8b-8192).
    *   API keys for these services must be configured through environment variables (`OPENAI_API_KEY`, `GROQ_API_KEY`).
