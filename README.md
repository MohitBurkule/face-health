# Face Health Analyzer

This project is a real-time face analysis tool that runs in your browser. It uses your webcam to detect facial landmarks and estimate vital signs and other health-related metrics.

**Live URL**: https://face-health.app.scot

## Features

- **Real-time Face Landmark Detection**: Utilizes MediaPipe's FaceLandmarker to detect 478 facial landmarks.
- **Remote Photoplethysmography (rPPG)**: Estimates heart rate, respiration rate, and heart rate variability (HRV) by analyzing subtle changes in skin color from the video feed.
- **Facial Fullness Estimation**: Provides a heuristic-based index of facial adiposity.
- **Eye & Drowsiness Metrics**: Tracks blink rate, PERCLOS (Percentage of Eye Closure), and yawn probability.
- **Facial Motion Tracking**: Measures relative landmark movement while attempting to cancel out rigid head motion.

## Tech Stack

- [Vite](https://vitejs.dev/)
- [React](https://reactjs.org/)
- [TypeScript](https://www.typescriptlang.org/)
- [Tailwind CSS](https://tailwindcss.com/)
- [shadcn/ui](https://ui.shadcn.com/)
- [MediaPipe](https://developers.google.com/mediapipe)

## Setup and Run

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```sh
    git clone <YOUR_GIT_URL>
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd face-health-analyzer
    ```

3.  **Install dependencies:**
    ```sh
    npm install
    ```

4.  **Start the development server:**
    ```sh
    npm run dev
    ```

The application will be available at `http://localhost:5173` (or another port if 5173 is busy).

## Disclaimer

This is an experimental application and is **not a medical device**. The estimations provided are for informational purposes only and should not be used for medical diagnosis.
