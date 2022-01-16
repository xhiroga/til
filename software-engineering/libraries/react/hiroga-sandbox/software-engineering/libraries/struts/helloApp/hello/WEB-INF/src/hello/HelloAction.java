package hello;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.struts.action.Action;
import org.apache.struts.action.ActionForm;
import org.apache.struts.action.ActionForward;
import org.apache.struts.action.ActionMapping;

import hello.form.HelloForm;

/**
* HelloAction.java
*/
public class HelloAction extends Action {

    Log log = LogFactory.getLog(HelloAction.class);

    public ActionForward execute(ActionMapping mapping,
                                    ActionForm form,
                                    HttpServletRequest request,
                                    HttpServletResponse response)
    throws Exception {

        request.setCharacterEncoding("Windows-31J"); 
        HelloForm helloForm = (HelloForm) form;
        log.info(" user = "+helloForm.getName());

        return mapping.findForward("success");
    }
}
