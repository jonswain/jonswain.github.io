---
layout: post
title: "Building a Traffic Reminder Widget"
date: 2025-05-01 12:00:00 +0100
categories:
  - general
---

Why spend a couple minutes doing something when you can spend an hour automating it? This is a WSL-based Python application to display Windows PowerShell notifications about driving conditions on your commute.

---

## Introduction

No data science or cheminformatics today!

I usually cycle or get public transport to work, but occasionally I do have to drive. The traffic in Tāmaki Makaurau (Auckland) is very variable, especially on rainy or windy days, so I often find myself checking Google Maps every 5 minutes after 4 pm to work out when I need to leave to get home. As with anything that I have to do repeatedly, [I decided  to automate it](https://xkcd.com/1205/).

My work computer runs Windows so I use [WSL](https://learn.microsoft.com/en-us/windows/wsl/about), which adds a few extra complications. All the code needed can be found in [this repository](https://github.com/jonswain/traffic-widget).

(I've also been experimenting with Google Gemini as a coding assistant. This wasn't completely [vibe-coding](https://en.wikipedia.org/wiki/Vibe_coding), but as this was a quick personal project, I was much less vigorous in checking the code it generated compared to more important work!)

All the Python code is kept in a file called `traffic-widget.py` which is stored on my WSL disk, I first needed to import the necessary libraries:

```python
import datetime
import json
import os
import subprocess

import requests
from dotenv import load_dotenv
```

## Getting the traffic data

Google maps doesn't seem to have a [completely free API](https://mapsplatform.google.com/pricing/). They do offer some free usage, but you still have to sign up and give credit card details, which is always a worry in case you accidentally go over the free limits (or accidentally leak your API key to the internet). TomTom on the other hand does have a [free API](https://www.tomtom.com/products/map-display-api/), you have to sign up and get an API key, and it comes with plenty of free requests. After signing up and getting and API key, I first created a TomTomAPI Python class, this uses the API key to make calls to the route calculation endpoint, providing the GPS coordinates for the start and end of your route, and returns the travel time.

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

To display the notifications I used BurntToast, a Windows PowerShell module for displaying Toast Notifications. To install, I needed to run PowerShell as admin and enter:

```PowerShell
Install-Module -Name BurntToast
```

BurntToast can be called from a PowerShell file (`.ps1`). The PowerShell file to display notifications is called `show_notitication.ps1` and is stored on my Windows disk. The `-Sound Alarm5` adds a sound to the notification and makes it last longer and `-AppLogo` gives icon beside the notification.

```ps1
param(
    [string]$Title,
    [string]$Message
)

$ImagePath = 'C:\Path\to\icon.jpg'

New-BurntToastNotification -Text $Title, $Message -AppLogo $ImagePath -Sound Alarm5
```

To allow the script to run, it may be necessary to run this command in PowerShell as admin:

```PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Running the PowerShell script from WSL

Back in the Python file (`traffic-widget.py`), I next needed a class to run the PowerShell file to display the notification.

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

This function formats the data from the TomTomAPI to be a more clear.

```python
def format_travel_time(label: str, travel_time: int | None) -> str:
    """Format the travel time into a readable string.

    Args:
        label (str): The label for the travel time (e.g., "Home").
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

To prevent sharing sentitive data such as my API key and home address, I stored these in a `.env` file. This also stores the paths to Windows PowerShell and the PowerShell script. Since this is working between two operating systems, the paths are slightly more complicated than usual. The PowerShell path is the path to the `powershell.exe` file on your Windows disk, from your Linux environment. This is usually something like: `/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe`. The Powershell script is in Windows format with escaped backslashes.

```env
API_KEY = "your_api_key"
WORK_LATITUDE = "work_latitude"
WORK_LONGITUDE = "work_longitude"
HOME_LATITUDE = "home_latitude"
HOME_LONGITUDE = "home_longitude"

POWERSHELL_PATH = "/mnt/c/path/to/powershell.exe"
POWERSHELL_SCRIPT = "C:\\windows\\path\\to\\show_notification.ps1"
```

## Creating a conda environment to run the script

To run the Python script I used a conda environment. It can be created and activated with:

```bash
conda env create -f environment.yml
conda activate traffic-widget
```

## Running the script

Finally I needed some Python code to run the whole process:

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

Manually running the script with `python traffic-widget.py` should cause the following pop-up:

![An example noficiation](/images/traffic_widget/traffic_notification.png){:class="img-responsive"}

## Setting up a cron job to automatically run the script

To make the script run automatically, back in WSL, I ran:

```bash
crontab -e
```

And added the details for the cron job. I wanted mine to run every 5 minutes from 4-5 pm on weekdays. 

```
0-55/5 16 * * 1-5 /home/<username>/miniconda3/envs/traffic-widget/bin/python /home/<username>/path/to/traffic-widget/traffic-widget.py
```