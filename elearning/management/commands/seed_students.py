"""
Management command to create student accounts.
Usage: python3 manage.py seed_students
"""
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from elearning.models import StudentProfile

STUDENTS = [
    {"email": "Emamul.embedded@gmail.com",       "first": "Emamul",   "last": "Embedded"},
    {"email": "sasidhar.madala2005@gmail.com",    "first": "Sasidhar", "last": "Madala"},
    {"email": "dhanushboyapati007@gmail.com",     "first": "Dhanush",  "last": "Boyapati"},
    {"email": "tonyjames1980@gmail.com",          "first": "Tony",     "last": "James"},
    {"email": "mohan.donthula28@gmail.com",       "first": "Mohan",    "last": "Donthula"},
    {"email": "riswanth22@gmail.com",             "first": "Riswanth", "last": ""},
]


class Command(BaseCommand):
    help = "Seed student user accounts from the predefined list"

    def handle(self, *args, **options):
        created_count = 0
        skipped_count = 0

        for s in STUDENTS:
            email = s["email"].lower()
            username = email.split("@")[0].replace(".", "_")

            if User.objects.filter(email__iexact=email).exists():
                self.stdout.write(self.style.WARNING(f"  SKIP  {email} (already exists)"))
                skipped_count += 1
                continue

            # Make username unique if collision
            base = username
            counter = 1
            while User.objects.filter(username=username).exists():
                username = f"{base}_{counter}"
                counter += 1

            # Default password = first part of email (before @), e.g. "emamul.embedded"
            password = email.split("@")[0]

            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                first_name=s["first"],
                last_name=s["last"],
            )

            # Create or update StudentProfile
            StudentProfile.objects.get_or_create(user=user)

            self.stdout.write(self.style.SUCCESS(
                f"  CREATED  {email}  (username={username}, password={password})"
            ))
            created_count += 1

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. Created: {created_count}, Skipped: {skipped_count}"
        ))
