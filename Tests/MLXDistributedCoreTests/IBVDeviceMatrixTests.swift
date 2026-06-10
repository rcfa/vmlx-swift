import Foundation
import MLXDistributedCore
import Testing

@Test func ibvMatrixAcceptsNullAndEmptySelfSlots() throws {
    let nullSelf = try IBVDeviceMatrix(jsonData: Data("""
    [
      [null, "rdma_en5"],
      ["rdma_en5", null]
    ]
    """.utf8))
    let emptySelf = IBVDeviceMatrix(rows: [
        ["", "mlx0"],
        ["mlx0", ""],
    ])

    #expect(nullSelf.isValid(worldSize: 2))
    #expect(emptySelf.isValid(worldSize: 2))
    #expect(nullSelf.findings(worldSize: 2).contains { $0.code == "ibv_matrix_valid" })
    #expect(emptySelf.findings(worldSize: 2).contains { $0.code == "ibv_matrix_valid" })
}

@Test func ibvMatrixRejectsMalformedShapeAndMissingPeerDevices() {
    let wrongWorldSize = IBVDeviceMatrix(rows: [
        [nil, "rdma_en5"],
        ["rdma_en5", nil],
    ])
    let notSquare = IBVDeviceMatrix(rows: [
        [nil, "rdma_en5"],
        ["rdma_en5"],
    ])
    let selfDevice = IBVDeviceMatrix(rows: [
        ["rdma_en5", "rdma_en5"],
        ["rdma_en5", nil],
    ])
    let missingPeer = IBVDeviceMatrix(rows: [
        [nil, ""],
        ["rdma_en5", nil],
    ])

    #expect(wrongWorldSize.findings(worldSize: 4).contains { $0.code == "ibv_matrix_world_size_mismatch" })
    #expect(notSquare.findings(worldSize: 2).contains { $0.code == "ibv_matrix_not_square" })
    #expect(selfDevice.findings(worldSize: 2).contains { $0.code == "ibv_matrix_self_slot_not_empty" })
    #expect(missingPeer.findings(worldSize: 2).contains { $0.code == "ibv_matrix_peer_device_missing" })
}
