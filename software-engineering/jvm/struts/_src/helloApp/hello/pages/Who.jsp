<%@ page contentType="text/html; charset=UTF-8" %>
<%@ taglib uri="http://struts.apache.org/tags-html"
  prefix="html" %>
<!DOCTYPE html>
<html:html>
  <head>
    <title>Who</title>
  </head>
  <html:form action="/hello">
    <table border="0">
      <html:errors/>
      <tr><td>
        あなたの名前は？<br>
        <html:text property="name" size="20" maxlength="30" />です。
      </td></tr>
      <tr><td>
        <html:submit value="OK" />
      </td></tr>
    </table>
  </html:form>
</html:html>
