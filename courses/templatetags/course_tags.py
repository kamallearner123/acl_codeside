from django import template

register = template.Library()

@register.filter
def split_comma(value):
    """Splits a comma-separated string into a list of stripped strings."""
    if not value:
        return []
    return [item.strip() for item in value.split(',')]
