from django import forms
from .models import Comment
from submissions.models import Submission


class SubmissionForm(forms.ModelForm):
    class Meta:
        model = Submission
        fields = ['code']
        widgets = {
            'code': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 20,
                'id': 'code-editor',
                'placeholder': 'Write your solution here...'
            })
        }


class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'Add a comment...',
                'class': 'form-control'
            }),
        }
