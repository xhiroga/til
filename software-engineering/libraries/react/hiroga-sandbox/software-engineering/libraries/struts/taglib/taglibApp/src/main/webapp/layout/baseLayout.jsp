<%@page contentType="text/html" pageEncoding="UTF-8"%>
<!DOCTYPE HTML>
<%@taglib uri="/WEB-INF/struts-tiles.tld" prefix="tiles" %>
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title><tiles:getAsString name="title" ignore="true" /></title>
    </head>
    <body>
        <tiles:insert attribute="header" ignore="true" />
        <tiles:insert attribute="body" />
    </body>
</html>
