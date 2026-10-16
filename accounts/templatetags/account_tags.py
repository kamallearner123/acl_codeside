from django import template
from allauth.socialaccount.models import SocialAccount
from accounts.models import UserProfile


register = template.Library()


@register.inclusion_tag("accounts/_user_avatar.html")
def user_avatar(user):
    avatar_url = ""
    profile = None

    try:
        profile = user.profile
    except UserProfile.DoesNotExist:
        pass

    if profile and profile.avatar:
        avatar_url = profile.avatar.url
    else:
        google_account = SocialAccount.objects.filter(
            user=user, provider="google"
        ).only("extra_data").first()
        if google_account:
            avatar_url = google_account.extra_data.get("picture", "")

    return {
        "avatar_url": avatar_url,
        "display_name": user.get_full_name() or user.get_username(),
    }
