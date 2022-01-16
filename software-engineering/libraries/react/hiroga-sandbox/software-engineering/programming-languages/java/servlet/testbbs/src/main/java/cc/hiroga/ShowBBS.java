package cc.hiroga;

import java.io.*;
import javax.servlet.http.*;

public class ShowBBS extends HttpServlet {

    @Override
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException{
        PrintWriter out = response.getWriter();
        out.println("<html>");
        out.println("Hi!");
        out.println("</html>");
    }
}