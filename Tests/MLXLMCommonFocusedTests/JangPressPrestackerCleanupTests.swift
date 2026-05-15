import Foundation
@testable import MLXLMCommon
import XCTest

final class JangPressPrestackerCleanupTests: XCTestCase {
    func testRegisteredEphemeralPrestackDirectoryCanBeRemovedBeforeProcessExit() throws {
        let overlay = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "jpress-ephemeral-prestack-\(UUID().uuidString)",
                isDirectory: true)
        try FileManager.default.createDirectory(
            at: overlay,
            withIntermediateDirectories: true)
        try Data("mapped already".utf8).write(
            to: overlay.appendingPathComponent("jangpress-prestacked.safetensors"))

        JangPressPrestacker.registerEphemeralPrestackDirectory(overlay)
        let removed = JangPressPrestacker.cleanupEphemeralPrestackDirectory(overlay)

        XCTAssertTrue(removed)
        XCTAssertFalse(FileManager.default.fileExists(atPath: overlay.path))
        XCTAssertFalse(JangPressPrestacker.cleanupEphemeralPrestackDirectory(overlay))
    }
}
