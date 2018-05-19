import React, { Component } from 'react';
import { Text } from 'react-native';
import {connect} from 'react-redux';
import {emailChanged, passwordChanged, loginUser} from '../actions';
import {Card, CardSection, Input, Button, Spinner} from './common';


class LoginForm extends Component{
  // text自身を渡すのにbind(this)を使ってる？
  onEmailChange(text){
    this.props.emailChanged(text)
  }

  onPasswordChange(text){
    this.props.passwordChanged(text)
  }

  onButtonPress() {
    const {email, password} = this.props;
    console.log('in onButtonPress', email, password)
    this.props.loginUser({email, password});
  }

  renderButton(){
    if (this.props.loading) {
      return <Spinner size="large" />;
    };
    return (
      <Button onPress = {this.onButtonPress.bind(this)} >
        Login
      </Button>
    )
  }// ヘルパーメソッド

  render(){
    return(
      <Card>
        <CardSection>
          <Input
            label='Email'
            placeholder="email@email.com"
            onChangeText={this.onEmailChange.bind(this)}
            value={this.props.email}
          />
        </CardSection>

        <CardSection>
          <Input
            secureTextEntry
            label='Password'
            placeholder="password"
            onChangeText={this.onPasswordChange.bind(this)}
            value={this.props.password}
          />
        </CardSection>

        <Text style={styles.errorTextStyle}>
          {this.props.error}
        </Text>

        <CardSection>
          {this.renderButton()}
        </CardSection>
      </Card>
    );
  };
}

const styles = {
  errorTextStyle :{
    fontSize: 20,
    alignSelf: 'center',
    color: 'red'
  }
}

// stateが変わったタイミングで再描画される
const mapStateToProps = ({auth}) => {
  const {email, password, error, loading } = auth;
  console.log("in connect, mapStateToProps", email, password, error)
  return { email, password, error, loading }
};

export default connect(mapStateToProps, {emailChanged, passwordChanged, loginUser})(LoginForm);
