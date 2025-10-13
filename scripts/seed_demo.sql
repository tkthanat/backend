USE attendance_db;

-- ประเภทผู้ใช้
INSERT INTO user_types(type_name) VALUES ('student'),('teacher'),('staff'),('housekeeping');

-- วิชา
INSERT INTO subjects(subject_name, section, schedule) VALUES
                                                          ('Computer Vision', 'A', 'Mon 09:00-12:00'),
                                                          ('Database Systems', 'B', 'Tue 10:00-12:00');

-- นักศึกษา (role = viewer เพราะไม่ล็อกอิน)
INSERT INTO users(student_code, name, role, user_type_id, subject_id)
SELECT '65010001','Alice Viewer','viewer', ut.user_type_id, s.subject_id
FROM user_types ut, subjects s
WHERE ut.type_name='student' AND s.subject_name='Computer Vision' AND s.section='A'
    LIMIT 1;

INSERT INTO users(student_code, name, role, user_type_id, subject_id)
SELECT '65010002','Bob Viewer','viewer', ut.user_type_id, s.subject_id
FROM user_types ut, subjects s
WHERE ut.type_name='student' AND s.subject_name='Computer Vision' AND s.section='A'
    LIMIT 1;

-- อาจารย์ (operator)
INSERT INTO users(student_code, name, role, user_type_id, subject_id)
SELECT NULL,'Dr. Teacher','operator', ut.user_type_id, s.subject_id
FROM user_types ut, subjects s
WHERE ut.type_name='teacher' AND s.subject_name='Computer Vision' AND s.section='A'
    LIMIT 1;

-- ใส่ embeddings แบบ mock (ไบต์ล้วนๆ พอทดสอบได้)
INSERT INTO user_embeddings(user_id, embedding_enc)
SELECT u.user_id, RANDOM_BYTES(512) FROM users u WHERE u.student_code IN ('65010001','65010002');

-- สร้าง log เข้า/ออกแบบเดโม (วันนี้)
SET @sid = (SELECT subject_id FROM subjects WHERE subject_name='Computer Vision' AND section='A' LIMIT 1);
SET @u1  = (SELECT user_id FROM users WHERE student_code='65010001' LIMIT 1);
SET @u2  = (SELECT user_id FROM users WHERE student_code='65010002' LIMIT 1);

INSERT INTO attendance_logs(user_id, subject_id, action, timestamp, camera_id, flags, confidence)
VALUES
    (@u1,@sid,'enter', NOW() - INTERVAL 110 MINUTE, 'entrance', JSON_OBJECT(), 0.92),
    (@u1,@sid,'exit',  NOW() - INTERVAL  10 MINUTE,  'exit',     JSON_OBJECT(), 0.88),
    (@u2,@sid,'enter', NOW() - INTERVAL  95 MINUTE, 'entrance', JSON_OBJECT(), 0.90),
    (@u2,@sid,'exit',  NOW() - INTERVAL  15 MINUTE, 'exit',     JSON_OBJECT(), 0.90);
