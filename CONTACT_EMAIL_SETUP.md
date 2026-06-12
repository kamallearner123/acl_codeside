# Contact Form Email System - Setup & Usage

## How It Works

Your contact form now has a **two-stage email system**:

### Stage 1: Immediate (On Form Submission)
1. User submits contact form
2. ✅ Contact data is saved to database
3. ✅ **Admin email sent immediately** to:
   - `info@aptcomputinglabs.com`
   - `kamal@aptcomputinglabs.com`
4. ✅ Success message shown to user (without mentioning confirmation email yet)

### Stage 2: Automatic (Within 24 Hours)
1. Run the management command to send confirmation emails
2. ✅ **Confirmation email sent to user** with their submission details
3. ✅ Tracked in database (marked as `user_email_sent = True`)

---

## How to Trigger Confirmation Emails

### Option 1: Manual (For Testing)
Run this command to send all pending confirmation emails:

```bash
python manage.py send_contact_confirmation_emails
```

Output example:
```
✓ Sent confirmation to user@example.com
✓ Sent confirmation to another@example.com

Total confirmation emails sent: 2
```

### Option 2: Automatic (Production - Every Hour)
Add a cron job to your server to run every hour:

```bash
# Every hour at the top of the hour
0 * * * * cd /path/to/acl_codeside && python manage.py send_contact_confirmation_emails
```

On PythonAnywhere:
1. Go to Web tab → Click your domain
2. Click "Schedule" 
3. Add task: `python /home/username/acl_codeside/manage.py send_contact_confirmation_emails`
4. Set to run hourly

---

## Database Fields Tracked

| Field | Purpose |
|-------|---------|
| `admin_email_sent` | True = email sent to your team |
| `user_email_sent` | True = confirmation sent to user |
| `created_at` | When form was submitted |

---

## Email Settings Configuration

✅ Already configured in `leetcode_clone/settings.py`:
- EMAIL_BACKEND: SMTP
- EMAIL_HOST: smtp.gmail.com
- EMAIL_HOST_USER: aptcomputinglabs@gmail.com
- EMAIL_HOST_PASSWORD: (app-specific password)
- DEFAULT_FROM_EMAIL: info@aptcomputinglabs.com

---

## Testing Locally

To test with console output instead of sending real emails:
```python
# In settings.py (development only)
EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
```

Change back to SMTP for production:
```python
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
```

---

## Files Modified

- ✅ `contact/views.py` - Now saves to database, sends admin email immediately
- ✅ `contact/models.py` - Added tracking fields
- ✅ `contact/management/commands/send_contact_confirmation_emails.py` - New command to send user confirmations
- ✅ `contact/migrations/0002_*.py` - Database changes
