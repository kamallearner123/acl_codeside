"""
Template filters that make blog article images render correctly and
responsively, no matter how the HTML in Post.content was produced
(hand-written, pasted from a rich-text editor, or converted from
Markdown/Word via pandoc).

Why this exists
----------------
Some existing posts (e.g. the AppArmor and CVE articles) were converted
from Word docs via pandoc, which bakes fixed, absolute-unit dimensions
directly onto each <img> as an inline style, e.g.:

    <img src="/static/blogs/cve/image1.png"
         style="width:6.26806in;height:3.275in" />

`.prose-custom img { max-width:100%; height:auto; }` (see
templates/blogs/detail.html) correctly caps the inline `width` on
narrow screens because `max-width` always wins over `width` for the
*same* box regardless of specificity. But the inline `height` is a
different property with nothing to clamp it, so on any screen narrower
than ~602px the image keeps its original fixed height while its width
shrinks - visibly squashing/stretching the image out of its real aspect
ratio.

Rather than hand-editing every historical post's stored HTML (fragile,
and would need to be redone for every future post that gets pasted in
with the same kind of markup), this filter neutralizes fixed-unit
width/height on <img> tags at render time, for every post, so the CSS
rule can size both dimensions consistently and the image always keeps
its true aspect ratio. It also adds native lazy-loading so article
images (usually below the fold) don't compete with the initial page
load.
"""
from django import template
from django.utils.safestring import mark_safe
from bs4 import BeautifulSoup

register = template.Library()

# Only these two CSS properties are stripped from an <img>'s inline
# style - anything else the content authored (if ever) is left alone.
_GEOMETRY_PROPS = ("width", "height")


@register.filter(name="render_blog_content", is_safe=True)
def render_blog_content(html):
    """
    Sanitize/normalize every <img> inside rich-text blog content so it
    renders responsively:

    - Drop any inline `width`/`height` (style or HTML attribute) that
      would otherwise override the responsive `.prose-custom img` CSS
      rule and distort the image's aspect ratio on smaller screens.
    - Add `loading="lazy"` + `decoding="async"` when not already set,
      so content images don't block the initial paint.

    Safe to run on every post, including ones with no images at all.
    """
    if not html:
        return html

    soup = BeautifulSoup(html, "html.parser")
    changed = False

    for img in soup.find_all("img"):
        changed = True

        # Drop legacy width="" / height="" HTML attributes.
        for attr in _GEOMETRY_PROPS:
            if img.has_attr(attr):
                del img[attr]

        # Strip only width/height declarations from an inline style,
        # keeping any other declaration the author may have added.
        style = img.get("style")
        if style:
            declarations = [d.strip() for d in style.split(";") if d.strip()]
            kept = [
                d for d in declarations
                if d.split(":", 1)[0].strip().lower() not in _GEOMETRY_PROPS
            ]
            if kept:
                img["style"] = "; ".join(kept)
            else:
                del img["style"]

        if not img.has_attr("loading"):
            img["loading"] = "lazy"
        if not img.has_attr("decoding"):
            img["decoding"] = "async"

    if not changed:
        return mark_safe(html)

    return mark_safe(str(soup))
