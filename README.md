# Apt Computing Labs - Django Web Application

A Django-based web application that replicates the core functionality of LeetCode, featuring Python coding challenges, user authentication, and code submission with automated testing.

## Features
- Problem constraints and hints
- Tag-based categorization

### 💻 Code Editor & Submission
- Problem acceptance rates
- User performance tracking
- Submission status tracking

- **Forms**: Django Crispy Forms
- **Icons**: Font Awesome
## Deploying to Azure

This project can be containerized and deployed to Azure Web App for Containers or Azure Container Apps.

Quick steps (Azure Web App for Containers):

1. Create an Azure Web App for Containers in the Azure Portal.
2. In GitHub repo settings -> Secrets, add:
	- AZURE_WEBAPP_NAME: your web app name
	- AZURE_PUBLISH_PROFILE: the publish profile XML content
	- DJANGO_SECRET_KEY: a secure secret
	- DATABASE_URL: your database connection string (e.g., Postgres)
	- GHCR_PAT: Personal Access Token with `write:packages` and `read:packages` to allow pushing to ghcr.io (if your org blocks the default GITHUB_TOKEN)
3. Push to `main` branch. The GitHub Actions workflow `.github/workflows/azure-deploy.yml` will build the image and deploy it.

Notes:
- The container uses Gunicorn and WhiteNoise to serve static files.
- For production, ensure secure values for `DJANGO_SECRET_KEY`, set `ALLOWED_HOSTS`, and use a managed Postgres. Replace the placeholder settings if you renamed the Django project package.


## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone <repository-url>
cd DeepCodeTest
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Database Setup
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Create Superuser
```bash
python manage.py createsuperuser
```

### 6. Load Sample Data
```bash
python manage.py load_sample_questions
```

### 7. Run Development Server
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` in your browser.

## Project Structure

```
apt_computing_labs/
├── accounts/                 # User authentication & profiles
│   ├── models.py            # UserProfile model
│   ├── views.py             # Auth views
│   ├── forms.py             # Registration & profile forms
│   └── urls.py              # Auth URLs
├── questions/               # Problem management
│   ├── models.py            # Question & Tag models
│   ├── views.py             # Problem listing & detail views
│   ├── forms.py             # Submission forms
│   ├── admin.py             # Admin interface
│   └── management/commands/ # Sample data loading
├── submissions/             # Code submission handling
│   ├── models.py            # Submission model
│   ├── views.py             # Submission views
│   └── admin.py             # Admin interface
├── templates/               # HTML templates
│   ├── base.html            # Base template
│   ├── home.html            # Homepage
│   ├── accounts/            # Auth templates
│   ├── questions/           # Problem templates
│   └── submissions/         # Submission templates
├── static/                  # Static files
│   ├── css/main.css         # Custom styles
│   └── js/main.js           # Custom JavaScript
├── leetcode_clone/          # Project settings
│   ├── settings.py          # Django settings
│   ├── urls.py              # Main URL configuration
│   └── wsgi.py              # WSGI configuration
└── manage.py                # Django management script
```

## Sample Problems Included

1. **Two Sum** (Easy) - Array, Hash Table
2. **Reverse String** (Easy) - String, Two Pointers
3. **Valid Parentheses** (Easy) - String, Stack
4. **Maximum Subarray** (Medium) - Array, Dynamic Programming
5. **Merge Two Sorted Lists** (Easy) - Recursion
6. **Climbing Stairs** (Easy) - Math, Dynamic Programming
7. **Binary Tree Inorder Traversal** (Easy) - Tree, Stack, Recursion
8. **Contains Duplicate** (Easy) - Array, Hash Table, Sorting

## Usage

### For Users
1. **Register/Login**: Create an account or login to existing account
2. **Browse Problems**: View problems filtered by difficulty
3. **Solve Problems**: Click on a problem to view details and submit solution
4. **Track Progress**: View submission history and statistics in your profile

### For Administrators
1. **Access Admin Panel**: Visit `/admin/` with superuser credentials
2. **Add Problems**: Create new coding problems with test cases
3. **Manage Users**: View and manage user accounts
4. **View Submissions**: Monitor all user submissions

## Key Models

### Question Model
```python
- title: Problem title
- slug: URL-friendly identifier
- description: Problem description
- difficulty: easy/medium/hard
- example_input/output: Sample test case
- template_code: Starting code template
- test_cases: JSON array of test cases
- constraints: Problem constraints
- hints: Problem hints
```

### Submission Model
```python
- user: Foreign key to User
- question: Foreign key to Question
- code: Submitted code
- status: accepted/wrong_answer/error/etc.
- test_cases_passed: Number of passed tests
- runtime: Execution time
- error_message: Error details if any
```

### UserProfile Model
```python
- user: One-to-one with User
- bio: User biography
- avatar: Profile picture
- github_username: GitHub handle
- problems_solved: Count of solved problems
- total_submissions: Total submission count
```

## Code Execution Security

⚠️ **Important**: The current code execution system is simplified for demonstration purposes. In production, you should:

1. Use containerized execution environments (Docker)
2. Implement proper sandboxing
3. Set execution time limits
4. Restrict system calls and imports
5. Use dedicated execution servers

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is for educational purposes. Please check the license file for details.

## Future Enhancements

- [ ] Discussion forums for each problem
- [ ] Editorial solutions
- [ ] Contest mode
- [ ] Advanced statistics and analytics
- [ ] Social features (following users, sharing solutions)
- [ ] Multiple programming language support
- [ ] Advanced code execution with proper sandboxing
- [ ] Real-time collaborative coding
- [ ] Company-specific problem tags
- [ ] Mock interview mode

## Support

For issues and questions, please create an issue in the repository or contact the development team.

---

**Note**: This is an educational project created to demonstrate Django web development skills and should not be used in production without proper security measures.
