from django.http import HttpResponse
from django.shortcuts import render_to_response
 
# Form
def search_form(request):
    return render_to_response('search_form.html')
 
# Get the request data

# def search(request):  
#     request.encoding='utf-8'
#     if 'q' in request.GET and request.GET['q']:
#         message = 'You are now searching: ' + request.GET['q']
#     else:
#         message = 'Error: EMPTY'
#     return HttpResponse(message)
def search(request):  
    request.encoding='utf-8'
    if 'q' in request.GET and request.GET['q']:
        # message = 'You are now searching: ' + request.GET['q'] + '\r\nsth'
        # return HttpResponse(message)
        response = HttpResponse()
        response.write('You are now searching: ' + request.GET['q'] + ' sth')
        response.write("<p>Here's the text of the Web page.</p>")
        response.write("<p>Here's another paragraph.</p>")
        return response
    else:
        return HttpResponse('Error: EMPTY')
