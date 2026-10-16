from django.test import TestCase
from django.contrib.auth.models import User
from django.template.loader import get_template
from allauth.socialaccount.models import SocialAccount
from accounts.templatetags.account_tags import user_avatar


class UserAvatarTests(TestCase):
    def test_navigation_templates_load(self):
        get_template("base.html")
        get_template("tailwind_base.html")

    def test_uses_google_profile_picture_when_no_uploaded_avatar_exists(self):
        user = User.objects.create_user(username="google-user")
        SocialAccount.objects.create(
            user=user,
            provider="google",
            uid="google-user-id",
            extra_data={"picture": "https://example.com/profile.jpg"},
        )

        avatar = user_avatar(user)

        self.assertEqual(avatar["avatar_url"], "https://example.com/profile.jpg")
        self.assertEqual(avatar["display_name"], "google-user")

    def test_falls_back_to_initial_when_no_profile_picture_exists(self):
        user = User.objects.create_user(username="google-user")

        avatar = user_avatar(user)

        self.assertEqual(avatar["avatar_url"], "")
        self.assertEqual(avatar["display_name"], "google-user")
