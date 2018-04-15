import React, {Component}from 'react';
import {View} from 'react-native';
import {}

class EmployeeForm extends Component {
  render(){
      retunr(
        <View>
          <CardSection>
            <Input
              label="Name"
              placeholder="Jane"
              value={this.props.name}
              onChangeText={value => this.props.employeeUpdate({prop: 'name', value})}
            />
          </CardSection>

          <CardSection>
            <Input
              label="Phone"
              placeholder="555-555-5555"
              value={this.props.phone}
              onChangeText={value => this.props.employeeUpdate({prop: 'phone', value})}
            />
          </CardSection>

          <CardSection style={{ flexDirection: 'column' }}>
            <Text style = {styles.pickerTextStyle}>Shift</Text>
            <Picker
              style={{flex:1}}
              selectedValue={this.props.shift}
              onValueChange={value => this.props.employeeUpdate({ prop: 'shift', value })}
            >
              <Picker.Item label="Monday" value="Monday" />
              <Picker.Item label="Tuesday" value="Tuesday" />
              <Picker.Item label="Wednesday" value="Wednesday" />
              <Picker.Item label="Thirsday" value="Thirsday" />
              <Picker.Item label="Friday" value="Friday" />
              <Picker.Item label="Saturday" value="Saturday" />
              <Picker.Item label="Sunday" value="Sunday" />
            </Picker>
          </CardSection>
        </View>
      )
  }
}


export default EmployeeForm;
