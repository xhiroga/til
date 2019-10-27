//
//  ViewController.swift
//  myFirstiBeacon
//
//  Created by Hiroaki Ogasawara on 2019/10/26.
//  Copyright © 2019 Hiroaki Ogasawara. All rights reserved.
//

import UIKit
import CoreBluetooth

class ViewController: UIViewController, CBPeripheralManagerDelegate {
    

    
    public override init(nibName nibNameOrNil: String?, bundle nibBundleOrNil: Bundle?){
        characteristic = CBMutableCharacteristic(
            type:CHARACTERISTIC_UUID,
            properties: CBCharacteristicProperties.read.union(CBCharacteristicProperties.notify),
            value: nil,
            permissions: CBAttributePermissions.readable
        )
        
        
        super.init(nibName: nibNameOrNil, bundle: nibBundleOrNil)
        
        peripheralManager = CBPeripheralManager(delegate: self, queue: nil)
        
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func peripheralManagerDidUpdateState(_ peripheral: CBPeripheralManager) {
        switch (peripheral.state) {
        case .poweredOn:
            print("Peripheral Manager state is on 🐵")
            ready = true
        default:
            print("Peripheral Manager state is off 🙉")
            ready = false
        }
        
    }
        
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
    }

    @IBAction func BeaconButton(_ sender: UIButton) {
        print("hi")
        peripheralManager.startAdvertising(["hi": "Hello"])
        
    }
    
}

