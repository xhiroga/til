package hello.form;

import org.apache.struts.action.ActionForm;

/**
* HelloForm.java
*/
public class HelloForm extends ActionForm {

    private String name;

    /**
    *
    * @return
    */
    public String getName() {
        return name;
    }
    /**
    *
    * @param name
    */
    public void setName(String name) {
        this.name = name;
    }
}
