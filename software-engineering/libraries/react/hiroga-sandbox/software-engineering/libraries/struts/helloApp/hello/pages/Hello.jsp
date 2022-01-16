<%@ page contentType="text/html; charset=UTF-8" %>
<%@ taglib uri="http://struts.apache.org/tags-html" 
  prefix="html" %>
<%@ taglib uri="http://struts.apache.org/tags-bean" 
  prefix="bean" %>
<!DOCTYPE html>
<html:html>
  <head>
    <title>Hello</title>
  </head>
  <html:form action="/hello">
    <table border="0">
      <tr><td>
        ようこそ<bean:write name="HelloForm" property="name" />さん！
      </td></tr>
    </table>
  </html:form>
</html:html>
