//
//  HomeViewController.swift
//  myFirstiBeacon
//
//  Created by Hiroaki Ogasawara on 2019/10/27.
//  Copyright © 2019 Hiroaki Ogasawara. All rights reserved.
//

import UIKit
import CoreBluetooth

class HomeViewController: UIViewController, CBPeripheralManagerDelegate {
    let CHARACTERISTIC_UUID = CBUUID(string: "9C76E7B3-AB72-41D0-96E0-438C75D7339A")
    let SERVICE_UUID = CBUUID(string: "19EFEDA2-99A4-4536-9EBE-7121FCE615C1")
    
    var manager: CBPeripheralManager!
    var characteristic: CBMutableCharacteristic!
    var service: CBMutableService!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.manager = CBPeripheralManager(delegate: self, queue: nil)
    }
    
    func peripheralManagerDidUpdateState(_ peripheral: CBPeripheralManager) {
        switch (peripheral.state) {
        case .poweredOn:
            print("Peripheral Manager state is on 🐵")
            self.addService()
        default:
            print("Peripheral Manager state is off 🙉")
        }
    }

    func addService(){
        self.service = CBMutableService(type: self.SERVICE_UUID, primary: true)
        self.characteristic = CBMutableCharacteristic(type: self.CHARACTERISTIC_UUID, properties: CBCharacteristicProperties.read, value: nil, permissions: CBAttributePermissions.readable)
        self.service.characteristics = [self.characteristic]
        
        self.manager.add(self.service)
    }

    @IBAction func startAdvertising(_ sender: Any) {
        self.manager.startAdvertising([CBAdvertisementDataServiceUUIDsKey: [self.service.uuid]])
    }
    
    func peripheralManagerDidStartAdvertising(_ peripheral: CBPeripheralManager, error: Error?) {
        if (error == nil){
            print("Advertising was started.")
        } else{
            print("Faild to start advertising.")
        }
    }
}
