import React, { Component } from 'react';

class Button extends Component {

    constructor(props) {
        super(props)
        this.state = { count: 1 }
    }

    render() {
        return (
            <div className="CountUp">
                <button>COUNT UP!!!</button>
            </div>
        );
    }
}

export default Button;