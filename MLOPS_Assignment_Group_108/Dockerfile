# Use the official Python 3.9 image as the base image
FROM python:3.9.13

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONPATH="/app"
ENV DATA_FILE_PATH="/app/Twitter_Sentiment_Indian_Election_2019/data/Twitter_Data.csv"
ENV PKL_FILE_PATH="/app/Twitter_Sentiment_Indian_Election_2019/pkl/best_model_twitter_senti.pkl"

# Copy the application files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the application will run on
EXPOSE 5001

# Command to run the application
CMD ["python", "/app/Twitter_Sentiment_Indian_Election_2019/src/main/app.py"]
