# Building a Traffic Reminder Widget

---
layout: post
title: "Building a Traffic Reminder Widget"
date: 2025-05-10 12:00:00 +0100
categories:
  - general
---

Abstract goes here

---

## Introduction

- I usually cycle to work, but occasionally I do have to drive
- Sadly traffic in Auckland can be awful, some days I find myself checking google maps every 5 minutes to see how the traffic is on my route home.
- Why not automate it and have a reminder pop up on my screen (https://xkcd.com/1205/)
- Sadly my work machine runs Windows, so I use WSL.
- I've also been experimenting with Google Gemini as a coding assistant. This wasn't completely vibe-coding, but as this was a quick personal project, I was much less vigarous in checking the code it generated compared to more important work!

First we need to import the necessary libraries:

```python
import datetime
import json
import os
import subprocess

import requests
from dotenv import load_dotenv
```

## Getting the traffic data

- Google maps doesn't seem to have a completely free API. You have to sign up and give credit card details, which is always a worry incase you accidentally go over the free limits (or accidentally leak your API key to the internet).
- TomTom does have a free API, you have to sign up and get an API key, and it comes with plenty of free requests.
- Created a TomTomAPI class, this uses your API key to make calls to the route calculation endpoint, providing the GPS coordinates for the start and end of your route, and returns the travel time.

```python
class TomTomAPI:
    """Encapsulates interactions with the TomTom Routing API."""

    BASE_URL = "https://api.tomtom.com/routing/1/calculateRoute"

    def __init__(self, api_key: str):
        """Initialize the TomTomAPI with the API key."""
        self.api_key = api_key

    def get_travel_time(
        self, start_lat: float, start_lon: float, end_lat: float, end_lon: float
    ) -> int | None:
        """Calculate the travel time between two points.

        Args:
            start_lat (float): Latitude of the starting point.
            start_lon (float): Longitude of the starting point.
            end_lat (float): Latitude of the destination point.
            end_lon (float): Longitude of the destination point.

        Returns:
            int: The travel time in seconds, or None if an error occurs.
        """
        start_point = f"{start_lat},{start_lon}"
        end_point = f"{end_lat},{end_lon}"
        url = f"{self.BASE_URL}/{start_point}:{end_point}/json?key={self.api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            travel_time_seconds = data["routes"][0]["summary"]["travelTimeInSeconds"]
            return travel_time_seconds

        except requests.exceptions.RequestException as e:
            print(f"TomTom API Error: {e}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            print(f"TomTom API Error: Invalid response format or missing data: {e}")
            return None
```

### Create the PowerShell script for the notification

Run powershell as admin

```PowerShell
Install-Module -Name BurntToast
```

Adding `-Sound Alarm5` also makes notification last longer. `-AppLogo` gives icon beside the notification

```ps1
param(
    [string]$Title,
    [string]$Message
)

$ImagePath = 'C:\Path\to\icon.jpg'

New-BurntToastNotification -Text $Title, $Message -AppLogo $ImagePath -Sound Alarm5
```

Allowing script to run (TODO: Some research on this)

```PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Running the PowerShell script from WSL

```python
class WindowsNotifier:
    """Handles displaying Windows toast notifications."""

    def __init__(self, powershell_path: str, powershell_script: str):
        """Initialize the WindowsNotifier with the paths to PowerShell."""
        self.powershell_path = powershell_path
        self.powershell_script = powershell_script

    def show_notification(self, title: str, message: str):
        """Show a Windows toast notification.

        Args:
            title (str): The title of the notification.
            message (str): The body of the notification.
        """
        try:
            subprocess.run(
                [
                    self.powershell_path,
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    self.powershell_script,
                    "-Title",
                    title,
                    "-Message",
                    message,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error showing notification: {e}")
            print(f"PowerShell Output:\n{e.stderr}")
        except FileNotFoundError:
            print(f"Error: PowerShell executable or script not found. Check the paths.")
```

### A helpful function to format the notification message

```python
def format_travel_time(label: str, travel_time: int | None) -> str:
    """Format the travel time into a readable string.

    Args:
        label (str): The label for the travel time (e.g., "Home", "Rowing").
        travel_time (int | None): The travel time in seconds, or None if an error occurred.

    Returns:
        str: The formatted travel time string.
    """
    now = datetime.datetime.now()
    arrival_time = (
        now + datetime.timedelta(seconds=travel_time) if travel_time else None
    )
    if travel_time is not None:
        return f"{label}: {travel_time / 60:.1f} minutes (arrive at {arrival_time.strftime('%H:%M')})"
    else:
        return f"{label}: ERROR minutes."
```

## Storing environment variables

Powershell path is something like "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
Powershell script is in Windows format with escaped backslashes.

```env
API_KEY = "your_api_keu"
WORK_LATITUDE = "work_latitude"
WORK_LONGITUDE = "work_longitude"
HOME_LATITUDE = "home_latitude"
HOME_LONGITUDE = "home_longitude"

POWERSHELL_PATH = "/mnt/c/path/to/powershell.exe"
POWERSHELL_SCRIPT = "C:\\windows\\path\\to\\show_notification.ps1"
```

## Running the script

```python
if __name__ == "__main__":
    load_dotenv()
    api_key = os.environ["API_KEY"]
    work_lat = float(os.environ["WORK_LATITUDE"])
    work_lon = float(os.environ["WORK_LONGITUDE"])
    home_lat = float(os.environ["HOME_LATITUDE"])
    home_lon = float(os.environ["HOME_LONGITUDE"])
    powershell_path = os.environ.get("POWERSHELL_PATH")
    powershell_script = os.environ.get("POWERSHELL_SCRIPT")
    if not powershell_script:
        print("Error: POWERSHELL_SCRIPT environment variable not set.")
        exit(1)

    tomtom_api = TomTomAPI(api_key)
    notifier = WindowsNotifier(powershell_path, powershell_script)

    home_travel_time = tomtom_api.get_travel_time(
        work_lat, work_lon, home_lat, home_lon
    )

    message_lines = [
        format_travel_time("Home", home_travel_time),
    ]
    notification_message = "\n".join(message_lines)

    notifier.show_notification("Driving times:", notification_message)
```

![An example noficiation](/images/traffic_widget/traffic_notification.png){:class="img-responsive"}