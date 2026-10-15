from allauth.account.adapter import DefaultAccountAdapter
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter


class NoLocalSignupAccountAdapter(DefaultAccountAdapter):
    """Only Google creates accounts through allauth; password sign-up stays in accounts.register."""

    def is_open_for_signup(self, request):
        return False


class GoogleSocialAccountAdapter(DefaultSocialAccountAdapter):
    """Google sign-in may create a new user on first login."""

    def is_open_for_signup(self, request, sociallogin):
        return True
