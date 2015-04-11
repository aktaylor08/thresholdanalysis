__author__ = 'ataylor'
#!/usr/bin/python

import smtplib


def mail_results(test_name, result, output):
    message = """From: From Person <from@fromdomain.com>
    To: Adam Taylor <aktaylor08@gmail.com>
    Subject: Test {:s} {:s}

    {:s}
    """.format(test_name, result, output)
    try:
        session = smtplib.SMTP('smtp.gmail.com',587)
        session.ehlo()
        session.starttls()
        session.ehlo()
        session.login("aktaylor08@gmail.com", "")
        session.sendmail("aktaylor08@gmail.com", "aktaylor08@gmail.com", message)
        session.quit()
    except smtplib.SMTPException:
        print "Error: unable to send email"
