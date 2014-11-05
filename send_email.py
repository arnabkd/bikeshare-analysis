import smtplib, datetime
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.utils import COMMASPACE, formatdate
from email import encoders
import os

def send_email(email_to, username, password, smtpserv, port, report, subject=None, files=None):
  """
  Send report 
  """
  print email_to, username, password
  current_time =datetime.datetime.now().strftime("%H:%M:%S on %d/%m/%Y")
  sender = username
  msg = MIMEMultipart()
  msg.attach(MIMEText(report))
  msg['Subject'] = ('Report generated at %s' % current_time) if subject is None else subject
  msg['From'] = sender
  msg['To'] = email_to

  if files is not None:
    for f in files:
      part = MIMEBase('application', "octet-stream")
      part.set_payload( open(f,"rb").read() )
      encoders.encode_base64(part)
      part.add_header('Content-Disposition', 'attachment; filename="{0}"'.format(os.path.basename(f)))
      msg.attach(part)
  
  s = smtplib.SMTP(smtpserv, port)
  s.ehlo()
  s.starttls()
  s.login(username, password)
  s.sendmail(sender, [email_to], msg.as_string())
  s.quit()
