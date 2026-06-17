# Upload this project to GitHub

## Option 1: Upload from the GitHub website

1. Create a new GitHub repository.
2. Download and unzip this project folder.
3. On GitHub, click **Add file → Upload files**.
4. Drag all project files and folders into the upload area.
5. Commit the upload.

## Option 2: Upload from Terminal

From inside the unzipped project folder, run:

```bash
git init
git add .
git commit -m "Initial commit: Bluesky padel vs tennis network analysis"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with your GitHub details.

## Before making the repo public

Check that you are allowed to publish the included Bluesky CSV files. They may contain public handles and post text.


## Included report

The folder now also includes `report/project_report.pdf`. Upload it together with the rest of the project files.
