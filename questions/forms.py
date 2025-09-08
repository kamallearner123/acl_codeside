from django import forms
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
