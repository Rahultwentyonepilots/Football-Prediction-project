#Code for Web Scraping using requests library 

import pandas as pd  # Import pandas for data manipulation
import time  # Import time for delays between retries
import random  # Import random for adding jitter in retry delays
import requests  # Import requests for handling HTTP requests
from requests.exceptions import RequestException  # Import RequestException for handling request errors

# List of URLs to scrape (each URL corresponds to a team's match logs(Scores and Fixtures))
urls = [
    'https://fbref.com/en/squads/b2b47a98/2021-2022/matchlogs/c9/schedule/Newcastle-United-Scores-and-Fixtures-Premier-League',
    'https://fbref.com/en/squads/47c64c55/2021-2022/matchlogs/c9/schedule/Crystal-Palace-Scores-and-Fixtures-Premier-League',
    'https://fbref.com/en/squads/cd051869/2021-2022/matchlogs/c9/schedule/Brentford-Scores-and-Fixtures-Premier-League',
]

# And for the below urls this is example: 

# List of URLs to scrape (each URL corresponds to a team's match logs(Shooting))
# urls =['https://fbref.com/en/squads/b8fd03ef/2023-2024/matchlogs/c9/shooting/Manchester-City-Match-Logs-Premier-League',
#         'https://fbref.com/en/squads/822bd0ba/2023-2024/matchlogs/c9/shooting/Liverpool-Match-Logs-Premier-League']


# List of URLs to scrape (each URL corresponds to a team's match logs(Passing))
# urls =['https://fbref.com/en/squads/822bd0ba/2023-2024/matchlogs/c9/passing/Liverpool-Match-Logs-Premier-League',
#         'https://fbref.com/en/squads/822bd0ba/2023-2024/matchlogs/c9/shooting/Liverpool-Match-Logs-Premier-League']


# List of URLs to scrape (each URL corresponds to a team's match logs(Goal and shot creation))
# urls =['https://fbref.com/en/squads/822bd0ba/2023-2024/matchlogs/c9/gca/Liverpool-Match-Logs-Premier-League',
#         'https://fbref.com/en/squads/822bd0ba/2023-2024/matchlogs/c9/gca/Liverpool-Match-Logs-Premier-League']






# Initialize a list to hold the dataframes extracted from each URL
dataframes = []

# Loop over each URL in the list
for url in urls:
    attempts = 0  # Initialize attempt counter
    max_attempts = 5  # Maximum number of attempts per URL
    success = False  # Success flag to indicate if data extraction was successful
    
    # Retry loop (up to max_attempts)
    while attempts < max_attempts and not success:
        try:
            # Read the HTML table with match logs from the URL
            df = pd.read_html(url, attrs={"id": "matchlogs_for"})[0]    #Replace with matchlogs_for_sh for shooting,
                                                                        #Replace with div_matchlogs_for for passing, 
                                                                        #Replace with div_matchlogs_for for goal and shot creation  
            dataframes.append(df)  # Append the dataframe to the list
            print(f"Successfully extracted data from {url}")  # Print success message
            success = True  # Set success flag to True
        except RequestException as e:
            # Handle request error and print the error message
            print(f"Failed to extract data from {url}: {e}")
            attempts += 1  # Increment the attempt counter
            
            # Handle "Too Many Requests" error (HTTP 429) with a 1-minute wait
            if "429" in str(e):
                wait_time = 60  # Wait time for rate-limiting
                print(f"Too many requests. Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)  # Pause execution for the wait time
            else:
                # Exponential backoff with jitter (random delay between retries)
                wait_time = (2 ** attempts) + random.uniform(0, 1)
                print(f"Retrying in {wait_time:.2f} seconds...")  # Print retry message
                time.sleep(wait_time)  # Pause execution for the wait time

# If any dataframes were successfully extracted, combine them
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)  # Concatenate dataframes
    print(combined_df.head())  # Display the first few rows of the combined dataframe
    combined_df.to_csv('combined_data.csv', index=False)  # Save the combined dataframe to a CSV file
else:
    print("No dataframes to concatenate")  # Print message if no data was scraped




