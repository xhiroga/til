//
//  AppDelegate.swift
//  myFirstiBeacon
//
//  Created by Hiroaki Ogasawara on 2019/10/26.
//  Copyright © 2019 Hiroaki Ogasawara. All rights reserved.
//

import UIKit
import CoreLocation

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate, CLLocationManagerDelegate {

    var locationManager: CLLocationManager?
    var lastProximity: CLProximity?

    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        // Override point for customization after application launch.
        
        let uuidString = "797B0EC3-18C0-4A95-8CB9-F3119B9D5994"
        let beaconRegionIdentifier = "hiroga.iBeacon"
        let beaconUUID: NSUUID = NSUUID(uuidString: uuidString)!
        
        let beaconRegion : CLBeaconRegion = CLBeaconRegion(proximityUUID: beaconUUID as UUID, identifier: beaconRegionIdentifier)
        
        print(beaconRegion)
        
        locationManager = CLLocationManager()
        
        if (locationManager!.responds(to: "requestAlwaysAuthorization")) {
            locationManager!.requestAlwaysAuthorization()
        }
        
        locationManager!.delegate = self
        locationManager!.pausesLocationUpdatesAutomatically = false
        
        locationManager!.startMonitoring(for: beaconRegion)
        
        // Start monitoring incase we have a beacon in our region
        locationManager!.startRangingBeacons(in: beaconRegion)
        locationManager!.startUpdatingLocation()
        
        return true
    }

    // MARK: UISceneSession Lifecycle

    func application(_ application: UIApplication, configurationForConnecting connectingSceneSession: UISceneSession, options: UIScene.ConnectionOptions) -> UISceneConfiguration {
        // Called when a new scene session is being created.
        // Use this method to select a configuration to create the new scene with.
        return UISceneConfiguration(name: "Default Configuration", sessionRole: connectingSceneSession.role)
    }

    func application(_ application: UIApplication, didDiscardSceneSessions sceneSessions: Set<UISceneSession>) {
        // Called when the user discards a scene session.
        // If any sessions were discarded while the application was not running, this will be called shortly after application:didFinishLaunchingWithOptions.
        // Use this method to release any resources that were specific to the discarded scenes, as they will not return.
    }
    
    func locationManager(_ manager: CLLocationManager, didEnterRegion region: CLRegion) {
        manager.startMonitoring(for: region)
        manager.startUpdatingLocation()
    }
    
    func locationManager(_ manager: CLLocationManager, didExitRegion region: CLRegion) {
        manager.startMonitoring(for: region)
        manager.startUpdatingLocation()
    }
    
    func locationManager(manager: CLLocationManager,
                         didRangeBeacons beacons: [CLBeacon],
                         inRegion region: CLBeaconRegion) {
        print("didRangeBeacons number of beacons found =\(beacons.count)")
    }

    
    

}

