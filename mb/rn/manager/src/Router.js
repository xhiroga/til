import React from 'react';
import { Scene, Router } from 'react-native-router-flux';
import {Actions} from 'react-native-router-flux';

import LoginForm from './components/LoginForm';
import EmployeeList from './components/EmployeeList';
import EmployeeCreate from './components/EmployeeCreate';


const RouterComponent = () => {

  // renderScene

  return (
    <Router sceneStyle={{ paddingTop: 65 }}>

      <Scene key="auth">
        <Scene key='login' component={LoginForm} title="Please Login" />
      </Scene>

      <Scene key="main">
        <Scene
          rightTitle='add'
          onRight={() => Actions.employeeCreate()}
          key='employeeList'
          component={EmployeeList}
          title="Employees"
          initial
        />

        <Scene key="employeeCreate" component={EmployeeCreate} title="Create Employee" />
      </Scene>

    </Router>
  );
};

export default RouterComponent;
